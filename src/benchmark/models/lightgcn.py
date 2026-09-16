"""
LightGCN (He et al., SIGIR 2020).
Light Graph Convolution Network for collaborative filtering.

Also includes LightGCN-SF: LightGCN with Side Feature injection
for content-augmented recommendation (the primary ablation host).
"""

import torch
import torch.nn as nn
from typing import Optional


class LightGCN(nn.Module):
    """
    LightGCN: ID-only collaborative filtering via graph convolution.
    Simplifies GCN by removing feature transformation and nonlinearity.
    Final embedding = mean of embeddings across all layers (including layer 0).
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 128,
        n_layers: int = 3,
        norm_adj: Optional[torch.sparse.FloatTensor] = None,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)

        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)

        # Normalized adjacency (set via set_adj or constructor)
        self.norm_adj = norm_adj

    def set_adj(self, norm_adj: torch.sparse.FloatTensor):
        """Set the normalized adjacency matrix (call after moving to device)."""
        self.norm_adj = norm_adj

    def _propagate(self):
        """
        LightGCN propagation: mean of layer-0 to layer-L embeddings.
        Returns (user_final, item_final) embeddings.
        """
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        emb_list = [all_emb]

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            emb_list.append(all_emb)

        # Mean pooling across layers
        stacked = torch.stack(emb_list, dim=0)
        final = stacked.mean(dim=0)

        user_final = final[:self.n_users]
        item_final = final[self.n_users:]
        return user_final, item_final

    def forward(self, users, pos_items, neg_items):
        """BPR forward: returns (pos_scores, neg_scores, reg_loss)."""
        user_final, item_final = self._propagate()

        u = user_final[users]
        p = item_final[pos_items]
        n = item_final[neg_items]

        pos_scores = (u * p).sum(dim=1)
        neg_scores = (u * n).sum(dim=1)

        # Regularize on initial embeddings (not propagated)
        u0 = self.user_emb(users)
        p0 = self.item_emb(pos_items)
        n0 = self.item_emb(neg_items)
        reg_loss = (u0.norm(2).pow(2) + p0.norm(2).pow(2) + n0.norm(2).pow(2)) / len(users)

        return pos_scores, neg_scores, reg_loss

    @torch.no_grad()
    def predict(self, user_ids):
        """Predict scores for all items. Returns (batch, n_items)."""
        user_final, item_final = self._propagate()
        u = user_final[user_ids]
        return u @ item_final.T


class LightGCNSF(nn.Module):
    """
    LightGCN with Side Features (LightGCN-SF).

    Injects pre-computed content features into item representations:
      item_repr = learned_emb(d) + MLP(content_features → d)

    The MLP-projected features are added to the initial item embeddings
    BEFORE graph propagation, so content information flows through the graph.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 128,
        n_layers: int = 3,
        feature_dim: int = 128,
        norm_adj: Optional[torch.sparse.FloatTensor] = None,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)

        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)

        # Feature projection MLP: content_features → embed_dim
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.norm_adj = norm_adj
        self.item_features = None  # Set via set_features()

    def set_adj(self, norm_adj: torch.sparse.FloatTensor):
        self.norm_adj = norm_adj

    def set_features(self, item_features: torch.Tensor):
        """Set item content features tensor. Shape: (n_items, feature_dim)."""
        self.item_features = item_features

    def _get_item_emb(self):
        """Item embedding = learned + MLP(content features)."""
        base = self.item_emb.weight
        if self.item_features is not None:
            projected = self.feature_proj(self.item_features)
            return base + projected
        return base

    def _propagate(self):
        """LightGCN propagation with content-augmented item embeddings."""
        item_emb = self._get_item_emb()
        all_emb = torch.cat([self.user_emb.weight, item_emb], dim=0)
        emb_list = [all_emb]

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            emb_list.append(all_emb)

        stacked = torch.stack(emb_list, dim=0)
        final = stacked.mean(dim=0)

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

        # Regularize initial embeddings + feature projection
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
