"""
SimGCL (Yu et al., SIGIR 2022): "Are Graph Augmentations Necessary? Simple Graph
Contrastive Learning for Recommendation".

An independent re-implementation of the method in the paper; its behaviour was
checked against the authors' public code (SELFRec, model/graph/SimGCL.py and
util/loss_torch.py). No code was copied. The points
below are the ones that were checked:

  - LightGCN propagation; the final embedding is the MEAN of layers 1..L. The
    input (layer-0) embedding is not part of it, in the clean pass or the
    perturbed one.
  - A perturbed pass adds, after each layer's propagation,
        sign(e) * normalize(U(0,1)) * eps
    so the noise has norm eps and keeps each coordinate's sign; the noisy
    embedding is what the next layer propagates.
  - The ranking (BPR) loss uses the clean pass. The contrastive loss is InfoNCE
    between TWO independently perturbed passes, over the batch's UNIQUE users and
    its UNIQUE positive items, with in-batch negatives, at temperature 0.2.
  - The constructor defaults, lambda = 0.5 and eps = 0.1, are the values in the
    authors' repository configuration (conf/SimGCL.yaml). They are not a
    setting for every dataset: the paper selects lambda per dataset, and so does
    this harness, on the validation split (scripts/contrastive_grid.py lists the
    grid and where each range comes from).

What stays the harness's own, as for every model it trains: embedding dimension,
layer count, batch size, optimiser, and the L2 term on the batch's input
embeddings (train.py scales it by the weight decay). Initialisation defaults to
the harness's Xavier-normal; init="uniform" is the authors' Xavier-uniform, which
the contrastive grid uses.

forward() returns (pos_scores, neg_scores, reg_loss, cl_loss). cl_loss is already
weighted by lambda, and train.py adds it as is (not scaled by the weight decay).

The InfoNCE log-sum-exp is computed in row chunks under activation
checkpointing: the result is identical, and peak memory is chunk x N instead of
N x N (N ~ 3e4 unique users per batch at this batch size).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional


class SimGCL(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 128,
        n_layers: int = 3,
        noise_eps: float = 0.1,
        cl_weight: float = 0.5,
        cl_temp: float = 0.2,
        norm_adj: Optional[torch.Tensor] = None,
        cl_chunk: int = 4096,
        init: str = "normal",   # "uniform" = Xavier-uniform, as the authors initialise
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.noise_eps = noise_eps
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

    @property
    def hparams(self) -> dict:
        """The method's own settings, recorded in results.json for provenance."""
        return {"model": "simgcl", "cl_weight": self.cl_weight, "noise_eps": self.noise_eps, "cl_temp": self.cl_temp, "init": self.init}

    def set_adj(self, norm_adj):
        self.norm_adj = norm_adj

    def _add_noise(self, emb):
        """sign(e) * normalize(U(0,1)) * eps, as in the authors' encoder."""
        noise = F.normalize(torch.rand_like(emb), dim=-1) * self.noise_eps
        return emb + torch.sign(emb) * noise

    def _propagate(self, perturb=False):
        """Mean of layers 1..L; layer 0 is not included (authors' encoder)."""
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        layers = []
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            if perturb:
                all_emb = self._add_noise(all_emb)
            layers.append(all_emb)
        final = torch.stack(layers, dim=0).mean(dim=0)
        return final[:self.n_users], final[self.n_users:]

    def _lse_block(self, block, view2):
        return torch.logsumexp(block @ view2.T / self.cl_temp, dim=-1)

    def _infonce_loss(self, view1, view2):
        """Mean InfoNCE over rows: -log softmax(view1 @ view2.T / t)[i, i].

        Equal to the authors' InfoNCE (cosine, in-batch negatives). The
        log-sum-exp is taken in row chunks; under autograd each chunk is
        checkpointed, so backward recomputes it instead of storing N x N.
        """
        view1 = F.normalize(view1, dim=-1)
        view2 = F.normalize(view2, dim=-1)
        pos = (view1 * view2).sum(dim=-1) / self.cl_temp
        lse = []
        for s in range(0, view1.shape[0], self.cl_chunk):
            block = view1[s:s + self.cl_chunk]
            if torch.is_grad_enabled() and block.requires_grad:
                lse.append(checkpoint(self._lse_block, block, view2, use_reentrant=False))
            else:
                lse.append(self._lse_block(block, view2))
        return (-pos + torch.cat(lse)).mean()

    def forward(self, users, pos_items, neg_items):
        # Ranking loss on the clean pass.
        user_final, item_final = self._propagate(perturb=False)
        u = user_final[users]
        p = item_final[pos_items]
        n = item_final[neg_items]
        pos_scores = (u * p).sum(dim=1)
        neg_scores = (u * n).sum(dim=1)

        # The harness's shared L2 term on the batch's input embeddings.
        u0 = self.user_emb(users)
        p0 = self.item_emb(pos_items)
        n0 = self.item_emb(neg_items)
        reg_loss = (u0.norm(2).pow(2) + p0.norm(2).pow(2) + n0.norm(2).pow(2)) / len(users)

        # Contrastive loss between two independently perturbed passes, over the
        # batch's unique users and unique positive items.
        u_idx = torch.unique(users)
        i_idx = torch.unique(pos_items)
        user_v1, item_v1 = self._propagate(perturb=True)
        user_v2, item_v2 = self._propagate(perturb=True)
        cl_loss = (self._infonce_loss(user_v1[u_idx], user_v2[u_idx])
                   + self._infonce_loss(item_v1[i_idx], item_v2[i_idx]))

        return pos_scores, neg_scores, reg_loss, self.cl_weight * cl_loss

    @torch.no_grad()
    def predict(self, user_ids):
        user_final, item_final = self._propagate(perturb=False)
        u = user_final[user_ids]
        return u @ item_final.T
