"""
LightGCL (Cai et al., ICLR 2023): "LightGCL: Simple Yet Effective Graph Contrastive
Learning for Recommendation".

An independent re-implementation of the method in the paper; its behaviour was
checked against the authors' public code (HKUDS/LightGCL, model.py and main.py). No code was copied. The points
below are the ones that were checked:

  - The main view is LightGCN propagation on the degree-normalised user-item graph
    (the same D_u^-1/2 R D_i^-1/2 block the harness builds); edge dropout is the
    authors' default of 0, so it is omitted.
  - The contrastive view comes from a rank-q SVD of that normalised matrix. Its layer
    l is built from the MAIN view's layer l-1:  G_u^l = U S V^T E_i^(l-1),
    G_i^l = V S U^T E_u^(l-1).
  - Both views are SUMMED over layers 0..L (layer 0 is the input embedding in both).
  - The contrastive loss is the authors' InfoNCE, WITHOUT cosine normalisation:
        -mean clamp(G[b] . E[b] / t, -5, 5) + mean log(sum_j exp(G[b] . E[j] / t) + 1e-8)
    with EVERY user (for users) or every item (for items) as a negative. Users are
    the batch's users and items are its positive and negative items, duplicates
    kept, as in the authors' loop.
  - The constructor defaults, lambda_1 = 0.2, tau = 0.2 and q = 5, are the defaults
    in the authors' code. Their README changes tau for some datasets; this harness
    selects tau on the validation split (scripts/contrastive_grid.py lists the grid
    and where each value comes from).

What stays the harness's own, as for every model it trains: embedding dimension,
layer count, batch size, optimiser, and the L2 term on the batch's input
embeddings (train.py scales it by the weight decay). Initialisation defaults to
the harness's Xavier-normal; init="uniform" is the authors' Xavier-uniform, which
the contrastive grid uses.

forward() returns (pos_scores, neg_scores, reg_loss, cl_loss). cl_loss is already
weighted by lambda_1, and train.py adds it as is (not scaled by the weight decay).

Negatives over every node are what make this expensive: at batch 32,768 the user
term alone is a 32,768 x 127,371 matrix (about 16 GB). The log-sum-exp is computed
in row chunks under activation checkpointing -- identical values, chunk x N memory.
"""

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
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
        norm_adj: Optional[torch.Tensor] = None,
        cl_chunk: int = 1024,    # rows per log-sum-exp chunk over all nodes
        init: str = "normal",   # "uniform" = Xavier-uniform, as the authors initialise
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.svd_q = svd_q
        self.cl_weight = cl_weight
        self.cl_temp = cl_temp
        self.cl_chunk = cl_chunk

        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)

        if init not in ("normal", "uniform"):
            raise ValueError(f"init must be 'normal' or 'uniform', got {init!r}")
        self.init = init
        xavier = nn.init.xavier_uniform_ if init == "uniform" else nn.init.xavier_normal_
        xavier(self.user_emb.weight)
        xavier(self.item_emb.weight)

        self.norm_adj = norm_adj
        # SVD-reconstructed adjacency (computed lazily in set_adj)
        self.svd_adj = None

    @property
    def hparams(self) -> dict:
        """The method's own settings, recorded in results.json for provenance."""
        return {"model": "lightgcl", "cl_weight": self.cl_weight, "cl_temp": self.cl_temp, "svd_q": self.svd_q, "init": self.init}

    def set_adj(self, norm_adj):
        """Set adjacency and compute SVD-based contrastive adjacency."""
        self.norm_adj = norm_adj
        self._compute_svd_adj()

    def _compute_svd_adj(self):
        """Rank-q SVD view of the user-item block, kept in FACTORED form.

        A dense (n_users+n_items)^2 adjacency, and a dense reconstruction of it,
        would each take 75.4 GB at ML-20M scale, and a full SVD of the dense
        127,371 x 9,906 block is not needed to keep five singular vectors.

        The factors are the same object the method is defined in terms of: a
        rank-q reconstruction of the normalised user-item block. Propagation
        multiplies by them directly (see `_propagate`), so the (n, n) matrix is
        never formed.

        The call is the authors' own (main.py: `torch.svd_lowrank(adj, q=svd_q)`,
        default two power iterations), and its random test matrix comes from the
        run's seeded generator, so each seed gets its own view, as each of the
        authors' runs did. That view is a rough approximation: on the ML-20M block,
        whose 5th and 6th singular values are close (0.462, 0.434), its rank-5
        reconstruction is 32-53% (relative Frobenius) from the exact truncation over
        five draws. Ten power iterations would give 18.7%. We keep the authors'
        call rather than improve on it.
        """
        adj = self.norm_adj.coalesce()
        device = adj.device

        # Factor on the CPU regardless of the training device: some accelerators
        # have partial sparse support, and svd_lowrank's get_approximate_basis
        # would densify the (n_users x n_users) basis there (about 60 GiB). On the
        # CPU the same call takes a couple of seconds, and only the q=5 factors
        # move to the device.
        idx, val = adj.indices().cpu(), adj.values().cpu()
        keep = (idx[0] < self.n_users) & (idx[1] >= self.n_users)
        block = torch.sparse_coo_tensor(
            torch.stack([idx[0][keep], idx[1][keep] - self.n_users]),
            val[keep], (self.n_users, self.n_items)).coalesce()

        # The draw comes from the global CPU generator, which set_seed() has seeded,
        # at the same point of construction on every launch; a resumed run rebuilds
        # the same view before its training RNG streams are restored.
        U, S, V = torch.svd_lowrank(block, q=self.svd_q)
        # Buffers, so .to(device) keeps moving them with the module afterwards.
        self.register_buffer("svd_u", U.to(device).contiguous(), persistent=False)
        self.register_buffer("svd_s", S.to(device).contiguous(), persistent=False)
        self.register_buffer("svd_v", V.to(device).contiguous(), persistent=False)
        self.svd_adj = None          # never materialised; see _propagate_svd

    def _propagate(self):
        """Both views, summed over layers 0..L (authors' model.py).

        Main view layer l: the normalised graph applied to layer l-1.
        SVD view layer l: the rank-q factors applied to the MAIN view's layer l-1.
        Returns (E_u, E_i, G_u, G_i).
        """
        e = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        e_sum = e
        g_sum = e
        s = self.svd_s.unsqueeze(1)
        for _ in range(self.n_layers):
            e_u, e_i = e[:self.n_users], e[self.n_users:]
            g = torch.cat([self.svd_u @ (s * (self.svd_v.t() @ e_i)),
                           self.svd_v @ (s * (self.svd_u.t() @ e_u))], dim=0)
            e = torch.sparse.mm(self.norm_adj, e)
            e_sum = e_sum + e
            g_sum = g_sum + g
        return (e_sum[:self.n_users], e_sum[self.n_users:],
                g_sum[:self.n_users], g_sum[self.n_users:])

    def _lse_block(self, block, all_emb):
        # log(sum_j exp(x_j) + 1e-8), exactly as the authors write it, but stable:
        # log(a + b) = logaddexp(log a, log b).
        lse = torch.logsumexp(block @ all_emb.T / self.cl_temp, dim=-1)
        return torch.logaddexp(lse, lse.new_tensor(math.log(1e-8)))

    def _cl_side(self, g_rows, e_rows, e_all):
        """One side of the authors' loss_s: -mean clamped positive + mean log-sum-exp."""
        pos = torch.clamp((g_rows * e_rows).sum(dim=-1) / self.cl_temp, -5.0, 5.0).mean()
        lse = []
        for s in range(0, g_rows.shape[0], self.cl_chunk):
            block = g_rows[s:s + self.cl_chunk]
            if torch.is_grad_enabled() and block.requires_grad:
                lse.append(checkpoint(self._lse_block, block, e_all, use_reentrant=False))
            else:
                lse.append(self._lse_block(block, e_all))
        return -pos + torch.cat(lse).mean()

    def forward(self, users, pos_items, neg_items):
        e_u, e_i, g_u, g_i = self._propagate()

        u = e_u[users]
        p = e_i[pos_items]
        n = e_i[neg_items]
        pos_scores = (u * p).sum(dim=1)
        neg_scores = (u * n).sum(dim=1)

        # The harness's shared L2 term on the batch's input embeddings.
        u0 = self.user_emb(users)
        p0 = self.item_emb(pos_items)
        n0 = self.item_emb(neg_items)
        reg_loss = (u0.norm(2).pow(2) + p0.norm(2).pow(2) + n0.norm(2).pow(2)) / len(users)

        # The authors' loss_s: users are the batch's users, items its positive and
        # negative items (duplicates kept); every node is a negative.
        iids = torch.cat([pos_items, neg_items])
        cl_loss = (self._cl_side(g_u[users], e_u[users], e_u)
                   + self._cl_side(g_i[iids], e_i[iids], e_i))

        return pos_scores, neg_scores, reg_loss, self.cl_weight * cl_loss

    def _propagate_main(self):
        """The main view alone, summed over layers 0..L."""
        e = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        e_sum = e
        for _ in range(self.n_layers):
            e = torch.sparse.mm(self.norm_adj, e)
            e_sum = e_sum + e
        return e_sum[:self.n_users], e_sum[self.n_users:]

    @torch.no_grad()
    def predict(self, user_ids):
        # The authors score with the main view only (model.py: E_u @ E_i.T), so
        # prediction needs no SVD factors: a model given just its adjacency, as the
        # evaluation from a saved checkpoint builds it, predicts exactly what
        # training measured.
        e_u, e_i = self._propagate_main()
        return e_u[user_ids] @ e_i.T
