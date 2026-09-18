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
        """Rank-q SVD view of the user-item block, kept in FACTORED form.

        This used to densify: `norm_adj.to_dense()` on the (n_users+n_items)^2
        adjacency, then `torch.zeros(n, n)` for the reconstruction. At ML-20M
        scale that is 75.4 GB each, so the model could not be constructed on an
        ordinary machine and M1d was the one configuration in the paper that a
        reader could not retrain at all -- its released numbers come from
        eval-only re-evaluation of a checkpoint. It also ran a FULL SVD of a
        127,371 x 9,906 dense matrix to keep five singular vectors.

        The factors are the same object the method is defined in terms of: a
        rank-q reconstruction of the normalised user-item block. Propagation
        multiplies by them directly (see `_propagate_svd`), so the (n, n) matrix
        is never formed. svd_lowrank is randomised, so the view differs slightly
        from an exact truncation; niter is raised for accuracy.
        """
        adj = self.norm_adj.coalesce()
        device = adj.device

        # Factor on the CPU regardless of the training device. MPS has partial
        # sparse support and svd_lowrank's get_approximate_basis densifies there,
        # which asks for a 60.44 GiB buffer (n_users^2 x 4 bytes) and aborts. On
        # the CPU the same call is a couple of seconds, and only the q=5 factors
        # move to the device.
        idx, val = adj.indices().cpu(), adj.values().cpu()
        keep = (idx[0] < self.n_users) & (idx[1] >= self.n_users)
        block = torch.sparse_coo_tensor(
            torch.stack([idx[0][keep], idx[1][keep] - self.n_users]),
            val[keep], (self.n_users, self.n_items)).coalesce()

        # svd_lowrank draws its random test matrix from the GLOBAL RNG, which
        # would make the contrastive view depend on the seed -- and on exactly how
        # much RNG was consumed before construction. The exact truncated SVD this
        # replaces was a deterministic property of the graph, identical for every
        # seed, so reproduce that: draw from a fixed local stream and put the
        # global one back, leaving the training stream untouched (which is what
        # lets a resumed run reproduce an uninterrupted one).
        _rng = torch.random.get_rng_state()
        try:
            torch.manual_seed(0)
            U, S, V = torch.svd_lowrank(block, q=self.svd_q, niter=10)
        finally:
            torch.random.set_rng_state(_rng)
        # Buffers, so .to(device) keeps moving them with the module afterwards.
        self.register_buffer("svd_u", U.to(device).contiguous(), persistent=False)
        self.register_buffer("svd_s", S.to(device).contiguous(), persistent=False)
        self.register_buffer("svd_v", V.to(device).contiguous(), persistent=False)
        self.svd_adj = None          # never materialised; see _propagate_svd

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

    def _propagate_svd(self):
        """Propagation on the rank-q SVD view, without ever forming it.

        The view is [[0, R], [R^T, 0]] with R = U diag(S) V^T, so for
        X = [X_u; X_i]:
            (view @ X)_users = U (S * (V^T X_i))
            (view @ X)_items = V (S * (U^T X_u))
        Both are rank-q products -- q is 5 -- so this is cheap and exact for
        the factored view.
        """
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        emb_list = [all_emb]
        s = self.svd_s.unsqueeze(1)

        for _ in range(self.n_layers):
            x_u, x_i = all_emb[:self.n_users], all_emb[self.n_users:]
            top = self.svd_u @ (s * (self.svd_v.t() @ x_i))
            bot = self.svd_v @ (s * (self.svd_u.t() @ x_u))
            all_emb = torch.cat([top, bot], dim=0)
            emb_list.append(all_emb)

        final = torch.stack(emb_list, dim=0).mean(dim=0)
        return final[:self.n_users], final[self.n_users:]

    def _infonce_loss(self, view1, view2):
        view1 = F.normalize(view1, dim=-1)
        view2 = F.normalize(view2, dim=-1)
        pos = (view1 * view2).sum(dim=-1) / self.cl_temp
        neg = view1 @ view2.T / self.cl_temp
        return (-pos + torch.logsumexp(neg, dim=-1)).mean()

    def forward(self, users, pos_items, neg_items):
        # Main view: propagate on original adjacency
        user_final, item_final = self._propagate(self.norm_adj)
        # Contrastive view: propagate on the factored SVD view
        user_svd, item_svd = self._propagate_svd()

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
