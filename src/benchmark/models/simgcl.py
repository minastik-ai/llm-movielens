"""
SimGCL (Yu et al., SIGIR 2022).
Simple contrastive learning for collaborative filtering.

Key insight: random noise perturbation on embeddings is sufficient for
contrastive augmentation — no graph augmentation needed.

Difference from XSimGCL: SimGCL uses same-layer contrast (perturbed vs clean
at the FINAL layer), while XSimGCL uses cross-layer contrast.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SimGCL(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 128,
        n_layers: int = 3,
        noise_eps: float = 0.1,
        cl_weight: float = 0.2,
        cl_temp: float = 0.2,
        norm_adj: Optional[torch.sparse.FloatTensor] = None,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.noise_eps = noise_eps
        self.cl_weight = cl_weight
        self.cl_temp = cl_temp

        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)

        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)

        self.norm_adj = norm_adj

    def set_adj(self, norm_adj):
        self.norm_adj = norm_adj

    def _add_noise(self, emb):
        noise = torch.rand_like(emb)
        noise = F.normalize(noise, dim=-1) * self.noise_eps
        return emb + noise

    def _propagate(self, perturb=False):
        """
        LightGCN propagation with optional noise perturbation.
        Returns mean-pooled final embeddings.
        """
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        emb_list = [all_emb]

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            if perturb:
                all_emb = self._add_noise(all_emb)
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
        # Clean view
        user_final, item_final = self._propagate(perturb=False)
        # Perturbed view (same-layer contrast)
        user_pert, item_pert = self._propagate(perturb=True)

        u = user_final[users]
        p = item_final[pos_items]
        n = item_final[neg_items]

        pos_scores = (u * p).sum(dim=1)
        neg_scores = (u * n).sum(dim=1)

        # Regularization
        u0 = self.user_emb(users)
        p0 = self.item_emb(pos_items)
        n0 = self.item_emb(neg_items)
        reg_loss = (u0.norm(2).pow(2) + p0.norm(2).pow(2) + n0.norm(2).pow(2)) / len(users)

        # Same-layer contrastive loss (clean final vs perturbed final)
        cl_loss_user = self._infonce_loss(user_final[users], user_pert[users])
        cl_loss_item = self._infonce_loss(item_final[pos_items], item_pert[pos_items])
        cl_loss = (cl_loss_user + cl_loss_item) * self.cl_weight

        return pos_scores, neg_scores, reg_loss + cl_loss

    @torch.no_grad()
    def predict(self, user_ids):
        user_final, item_final = self._propagate(perturb=False)
        u = user_final[user_ids]
        return u @ item_final.T
