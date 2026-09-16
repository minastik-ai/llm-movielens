"""
XSimGCL (Yu et al., TKDE 2023).
Cross-layer contrastive learning for collaborative filtering.

Adds noise-based contrastive regularization to LightGCN without graph augmentation.
Uses cross-layer contrast: compares embeddings from different propagation layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class XSimGCL(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 128,
        n_layers: int = 3,
        cl_layer: int = 2,       # which layer to use for contrastive view
        noise_eps: float = 0.1,  # noise perturbation magnitude
        cl_weight: float = 0.2,  # contrastive loss weight
        cl_temp: float = 0.2,    # InfoNCE temperature
        norm_adj: Optional[torch.sparse.FloatTensor] = None,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.cl_layer = cl_layer
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
        """Add uniform random noise for contrastive perturbation."""
        noise = torch.rand_like(emb)
        noise = F.normalize(noise, dim=-1) * self.noise_eps
        return emb + noise

    def _propagate(self, perturb=False):
        """
        LightGCN propagation. If perturb=True, adds noise at each layer
        for contrastive view.
        Returns: (final_emb, cl_layer_emb) — final is mean-pooled, cl_layer is from specific layer.
        """
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        emb_list = [all_emb]
        cl_emb = None

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            if perturb:
                all_emb = self._add_noise(all_emb)
            emb_list.append(all_emb)
            if layer == self.cl_layer:
                cl_emb = all_emb

        final = torch.stack(emb_list, dim=0).mean(dim=0)
        user_final = final[:self.n_users]
        item_final = final[self.n_users:]

        if cl_emb is not None:
            user_cl = cl_emb[:self.n_users]
            item_cl = cl_emb[self.n_users:]
        else:
            user_cl = user_final
            item_cl = item_final

        return user_final, item_final, user_cl, item_cl

    def _infonce_loss(self, view1, view2):
        """InfoNCE contrastive loss between two views of the same nodes."""
        view1 = F.normalize(view1, dim=-1)
        view2 = F.normalize(view2, dim=-1)
        pos = (view1 * view2).sum(dim=-1) / self.cl_temp
        # Use all nodes in batch as negatives
        neg = view1 @ view2.T / self.cl_temp
        loss = -pos + torch.logsumexp(neg, dim=-1)
        return loss.mean()

    def forward(self, users, pos_items, neg_items):
        # Main view (no perturbation)
        user_final, item_final, _, _ = self._propagate(perturb=False)
        # Contrastive view (with perturbation)
        _, _, user_cl, item_cl = self._propagate(perturb=True)

        u = user_final[users]
        p = item_final[pos_items]
        n = item_final[neg_items]

        pos_scores = (u * p).sum(dim=1)
        neg_scores = (u * n).sum(dim=1)

        # BPR reg
        u0 = self.user_emb(users)
        p0 = self.item_emb(pos_items)
        n0 = self.item_emb(neg_items)
        reg_loss = (u0.norm(2).pow(2) + p0.norm(2).pow(2) + n0.norm(2).pow(2)) / len(users)

        # Cross-layer contrastive loss on batch users/items
        # Compare main view final embeddings with perturbed cl_layer embeddings
        cl_loss_user = self._infonce_loss(user_final[users], user_cl[users])
        cl_loss_item = self._infonce_loss(item_final[pos_items], item_cl[pos_items])
        cl_loss = (cl_loss_user + cl_loss_item) * self.cl_weight

        # Return combined: BPR component + CL component
        # Training loop uses: bpr_loss(pos, neg) + wd * reg + cl_loss
        return pos_scores, neg_scores, reg_loss + cl_loss

    @torch.no_grad()
    def predict(self, user_ids):
        user_final, item_final, _, _ = self._propagate(perturb=False)
        u = user_final[user_ids]
        return u @ item_final.T
