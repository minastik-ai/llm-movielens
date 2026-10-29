"""
BPR Matrix Factorization (Rendle et al., UAI 2009).
Classic pairwise collaborative filtering baseline.
"""

import torch
import torch.nn as nn


class BPRMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, embed_dim: int = 128):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim

        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)

        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)

    def forward(self, users, pos_items, neg_items):
        """
        BPR forward pass.
        Returns: (pos_scores, neg_scores, reg_loss)
        """
        u = self.user_emb(users)
        p = self.item_emb(pos_items)
        n = self.item_emb(neg_items)

        pos_scores = (u * p).sum(dim=1)
        neg_scores = (u * n).sum(dim=1)

        reg_loss = (u.norm(2).pow(2) + p.norm(2).pow(2) + n.norm(2).pow(2)) / len(users)

        return pos_scores, neg_scores, reg_loss

    @torch.no_grad()
    def predict(self, user_ids):
        """
        Predict scores for all items for given users.
        Returns: (batch_size, n_items)
        """
        u = self.user_emb(user_ids)  # (batch, dim)
        items = self.item_emb.weight  # (n_items, dim)
        return u @ items.T
