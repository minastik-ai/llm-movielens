"""
KAR — Knowledge-Augmented Recommendation (Xi et al., KDD 2024).
"Towards open-world recommendation with knowledge augmentation from LLMs."

Integrates LLM-generated knowledge features into a LightGCN backbone
via a hybrid-expert adapter: a mixture-of-experts (MoE) module that
learns to combine CF embeddings with knowledge features through
multiple expert MLPs gated by the CF representation.

Simplified for benchmark comparability: uses our pre-computed LLM
features as the "knowledge" input (rather than generating knowledge
on-the-fly from an LLM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class KAR(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 128,
        n_layers: int = 3,
        feature_dim: int = 128,
        n_experts: int = 4,       # number of expert networks
        norm_adj: Optional[torch.sparse.FloatTensor] = None,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_experts = n_experts

        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)

        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)

        # Hybrid-expert adapter: multiple expert MLPs + gating network
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            for _ in range(n_experts)
        ])

        # Gating network: CF embedding → expert weights
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, n_experts),
        )

        self.norm_adj = norm_adj
        self.item_features = None

    def set_adj(self, norm_adj):
        self.norm_adj = norm_adj

    def set_features(self, item_features: torch.Tensor):
        """Set item knowledge features. Shape: (n_items, feature_dim)."""
        self.item_features = item_features

    def _get_item_emb(self):
        """
        Hybrid-expert adapter: item_emb + MoE(knowledge_features).
        Gate weights are derived from the learned CF embedding, so the
        model learns which experts to trust based on the item's CF profile.
        """
        base = self.item_emb.weight
        if self.item_features is not None:
            # Compute expert outputs: list of (n_items, embed_dim)
            expert_out = torch.stack(
                [expert(self.item_features) for expert in self.experts],
                dim=1,
            )  # (n_items, n_experts, embed_dim)

            # Gate: CF embedding → softmax expert weights
            gate_scores = F.softmax(self.gate(base), dim=-1)  # (n_items, n_experts)
            gate_scores = gate_scores.unsqueeze(-1)  # (n_items, n_experts, 1)

            # Weighted sum of expert outputs
            knowledge_emb = (gate_scores * expert_out).sum(dim=1)  # (n_items, embed_dim)
            return base + knowledge_emb
        return base

    def _propagate(self):
        """LightGCN propagation with knowledge-augmented item embeddings."""
        item_emb = self._get_item_emb()
        all_emb = torch.cat([self.user_emb.weight, item_emb], dim=0)
        emb_list = [all_emb]

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            emb_list.append(all_emb)

        final = torch.stack(emb_list, dim=0).mean(dim=0)
        user_final = final[:self.n_users]
        item_final = final[self.n_users:]
        return user_final, item_final

    def forward(self, users, pos_items, neg_items):
        user_final, item_final = self._propagate()

        u = user_final[users]
        p = item_final[pos_items]
        n = item_final[neg_items]

        pos_scores = (u * p).sum(dim=1)
        neg_scores = (u * n).sum(dim=1)

        # Regularization on initial embeddings
        u0 = self.user_emb(users)
        p0 = self.item_emb(pos_items)
        n0 = self.item_emb(neg_items)
        reg_loss = (u0.norm(2).pow(2) + p0.norm(2).pow(2) + n0.norm(2).pow(2)) / len(users)

        return pos_scores, neg_scores, reg_loss

    @torch.no_grad()
    def predict(self, user_ids):
        user_final, item_final = self._propagate()
        u = user_final[user_ids]
        return u @ item_final.T
