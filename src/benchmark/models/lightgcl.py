"""
LightGCL (Cai et al., ICLR 2023).
Simple yet effective graph contrastive learning for recommendation.

Key insight: uses SVD decomposition of the adjacency matrix to create
informative contrastive views, replacing random noise perturbation.
The low-rank SVD reconstruction captures global collaborative patterns
that serve as a structurally meaningful augmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LightGCL(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 128,
        n_layers: int = 3,
        svd_q: int = 5,         # SVD rank for contrastive view
        cl_weight: float = 0.2,  # contrastive loss weight
        cl_temp: float = 0.2,    # InfoNCE temperature
        norm_adj: Optional[torch.sparse.FloatTensor] = None,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.svd_q = svd_q
        self.cl_weight = cl_weight
        self.cl_temp = cl_temp

        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)

        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)

        self.norm_adj = norm_adj
        # SVD-reconstructed adjacency (computed lazily in set_adj)
        self.svd_adj = None

    def set_adj(self, norm_adj):
        """Set adjacency and compute SVD-based contrastive adjacency."""
        self.norm_adj = norm_adj
        self._compute_svd_adj()

    def _compute_svd_adj(self):
        """Compute low-rank SVD approximation of the adjacency matrix."""
        adj = self.norm_adj
        # Convert to dense for SVD (on the user-item bipartite subblock)
        # Extract the user→item block: top-right of shape (n_users, n_items)
        adj_dense = adj.to_dense()
        ui_block = adj_dense[:self.n_users, self.n_users:]

        # Truncated SVD
        U, S, Vh = torch.linalg.svd(ui_block, full_matrices=False)
        U_q = U[:, :self.svd_q]
        S_q = S[:self.svd_q]
        Vh_q = Vh[:self.svd_q, :]

        # Reconstruct: user→item block approximation
        ui_svd = U_q @ torch.diag(S_q) @ Vh_q

        # Rebuild full symmetric adjacency from reconstructed block
        # Original structure: [[0, R], [R^T, 0]] (normalized)
        n = self.n_users + self.n_items
        svd_dense = torch.zeros(n, n, device=adj.device)
        svd_dense[:self.n_users, self.n_users:] = ui_svd
        svd_dense[self.n_users:, :self.n_users] = ui_svd.T

        self.svd_adj = svd_dense.to_sparse()

    def _propagate(self, adj):
        """LightGCN propagation with given adjacency."""
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        emb_list = [all_emb]

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(adj, all_emb)
            emb_list.append(all_emb)

        final = torch.stack(emb_list, dim=0).mean(dim=0)
        user_final = final[:self.n_users]
        item_final = final[self.n_users:]
        return user_final, item_final

    def _infonce_loss(self, view1, view2):
        view1 = F.normalize(view1, dim=-1)
        view2 = F.normalize(view2, dim=-1)
        pos = (view1 * view2).sum(dim=-1) / self.cl_temp
        neg = view1 @ view2.T / self.cl_temp
        return (-pos + torch.logsumexp(neg, dim=-1)).mean()

    def forward(self, users, pos_items, neg_items):
        # Main view: propagate on original adjacency
        user_final, item_final = self._propagate(self.norm_adj)
        # Contrastive view: propagate on SVD-reconstructed adjacency
        user_svd, item_svd = self._propagate(self.svd_adj)

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

        # Contrastive loss: original view vs SVD view
        cl_loss_user = self._infonce_loss(user_final[users], user_svd[users])
        cl_loss_item = self._infonce_loss(item_final[pos_items], item_svd[pos_items])
        cl_loss = (cl_loss_user + cl_loss_item) * self.cl_weight

        return pos_scores, neg_scores, reg_loss + cl_loss

    @torch.no_grad()
    def predict(self, user_ids):
        user_final, item_final = self._propagate(self.norm_adj)
        u = user_final[user_ids]
        return u @ item_final.T
