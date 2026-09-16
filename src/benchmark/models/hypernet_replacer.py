"""
HypernetReplacer (R3) — second instantiation of the replacer paradigm class
for the replacer triangulation experiment.

Architectural difference from R2 (KAR-style HEA):
- R2: item_embed[i] = learned_ID_embed[i] + MoE_over_content(features[i])
       (CF embedding is augmented; gating uses CF embedding as input)
- R3: item_embed[i] = hypernet(features[i])
       (pure content→embedding; NO learned per-item ID capacity at all)

R3 is maximally distant from R2 within the replacer class:
  - No experts, no MoE
  - No gating
  - No CF-derived input to the content-mapping
  - Items have ZERO learned ID-side capacity (only user side does)

If R3 ALSO collapses on Amazon-Books, the replacer-class claim triangulates:
two structurally-different replacers fail in the same density regime, so the
failure is class-level, not instantiation-specific.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class HypernetReplacer(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 128,
        n_layers: int = 3,
        feature_dim: int = 128,
        hidden: int = 256,
        norm_adj: Optional[torch.sparse.FloatTensor] = None,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        # Users keep ID-side capacity (matches KAR / LightGCN convention)
        self.user_emb = nn.Embedding(n_users, embed_dim)
        nn.init.xavier_normal_(self.user_emb.weight)

        # Items are PURELY content-derived. No nn.Embedding for items.
        self.hypernet = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )
        # Xavier init on the linear layers
        for m in self.hypernet:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        self.norm_adj = norm_adj
        self.item_features: Optional[torch.Tensor] = None

    def set_adj(self, norm_adj):
        self.norm_adj = norm_adj

    def set_features(self, item_features: torch.Tensor):
        """Set item content features. Shape: (n_items, feature_dim)."""
        self.item_features = item_features

    def _get_item_emb(self) -> torch.Tensor:
        """Pure content→embedding mapping. No CF-side capacity."""
        assert self.item_features is not None, "Call set_features() first"
        return self.hypernet(self.item_features)  # (n_items, embed_dim)

    def _propagate(self):
        """LightGCN propagation with hypernet-derived item embeddings."""
        item_emb = self._get_item_emb()
        all_emb = torch.cat([self.user_emb.weight, item_emb], dim=0)
        emb_list = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            emb_list.append(all_emb)
        final = torch.stack(emb_list, dim=0).mean(dim=0)
        user_final = final[: self.n_users]
        item_final = final[self.n_users :]
        return user_final, item_final

    def forward(self, users, pos_items, neg_items):
        user_final, item_final = self._propagate()
        u = user_final[users]
        p = item_final[pos_items]
        n = item_final[neg_items]
        pos_scores = (u * p).sum(dim=1)
        neg_scores = (u * n).sum(dim=1)

        # Regularization: user_emb (ID side) + hypernet output norms (content side)
        # Mirrors KAR's "regularize the input embeddings" but adapted to no-ID-emb-for-items.
        u0 = self.user_emb(users)
        # For the item side, regularize the hypernet outputs at the queried items
        # (this is the cleanest analog to "regularize the item embedding")
        p0 = self._get_item_emb()[pos_items]
        n0 = self._get_item_emb()[neg_items]
        reg_loss = (
            u0.norm(2).pow(2) + p0.norm(2).pow(2) + n0.norm(2).pow(2)
        ) / len(users)

        return pos_scores, neg_scores, reg_loss

    @torch.no_grad()
    def predict(self, user_ids):
        user_final, item_final = self._propagate()
        u = user_final[user_ids]
        return u @ item_final.T
