"""
PyTorch Dataset for BPR training and evaluation on ML-20M.

Provides:
  - BPRDataset: yields (user, pos_item, neg_item) triplets for BPR training
  - InteractionData: loads train/val/test splits, builds adjacency, provides user histories
"""

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional
import scipy.sparse as sp

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, BATCH_SIZE, NUM_NEGATIVES


class InteractionData:
    """
    Loads processed train/val/test splits and provides:
      - user/item counts
      - user interaction histories (for negative sampling and evaluation)
      - sparse adjacency matrix (for GNN models)
    """

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir

        # Load splits
        self.train_df = pd.read_csv(data_dir / "train.csv")
        self.val_df = pd.read_csv(data_dir / "val.csv")
        self.test_df = pd.read_csv(data_dir / "test.csv")

        # Load stats
        with open(data_dir / "stats.json") as f:
            self.stats = json.load(f)

        self.n_users = self.stats["n_users"]
        self.n_items = self.stats["n_items"]

        # Load ID mappings
        with open(data_dir / "item_map.json") as f:
            self.item_map = {int(k): v for k, v in json.load(f).items()}
        self.item_map_inv = {v: k for k, v in self.item_map.items()}

        # Build user interaction sets
        self.train_user_items = self._build_user_items(self.train_df)
        self.val_user_items = self._build_user_items(self.val_df)
        self.test_user_items = self._build_user_items(self.test_df)

        # All items set for negative sampling
        self.all_items = set(range(self.n_items))

    def _build_user_items(self, df: pd.DataFrame) -> Dict[int, Set[int]]:
        """Build {user_id: set(item_ids)} from a DataFrame."""
        user_items = {}
        for uid, group in df.groupby("userId"):
            user_items[int(uid)] = set(group["movieId"].tolist())
        return user_items

    def get_sparse_adj(self) -> sp.coo_matrix:
        """
        Build sparse bipartite adjacency matrix for LightGCN.
        Shape: (n_users + n_items, n_users + n_items)
        """
        users = self.train_df["userId"].values
        items = self.train_df["movieId"].values + self.n_users  # offset item IDs

        # Symmetric: user→item and item→user edges
        rows = np.concatenate([users, items])
        cols = np.concatenate([items, users])
        vals = np.ones(len(rows), dtype=np.float32)

        adj = sp.coo_matrix(
            (vals, (rows, cols)),
            shape=(self.n_users + self.n_items, self.n_users + self.n_items),
        )
        return adj

    def get_norm_adj(self) -> torch.sparse.FloatTensor:
        """
        Normalized adjacency: D^{-1/2} A D^{-1/2} as sparse torch tensor.
        Used by LightGCN for message passing.
        """
        adj = self.get_sparse_adj()

        # Degree normalization
        rowsum = np.array(adj.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat = sp.diags(d_inv_sqrt)

        norm_adj = d_mat @ adj @ d_mat
        norm_adj = norm_adj.tocoo()

        indices = torch.LongTensor(np.stack([norm_adj.row, norm_adj.col]))
        values = torch.FloatTensor(norm_adj.data)
        shape = torch.Size(norm_adj.shape)

        return torch.sparse_coo_tensor(indices, values, shape).coalesce()

    def get_train_pairs(self) -> np.ndarray:
        """Return (user, item) positive pairs from training data."""
        return self.train_df[["userId", "movieId"]].values


class BPRDataset(Dataset):
    """
    BPR triplet dataset: yields (user, pos_item, neg_item).
    Negative items are sampled uniformly from items the user has NOT interacted with.
    """

    def __init__(self, interaction_data: InteractionData):
        self.data = interaction_data
        self.pairs = self.data.get_train_pairs()
        self.n_items = self.data.n_items
        self.train_user_items = self.data.train_user_items

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        user, pos_item = self.pairs[idx]

        # Sample negative item (not in user's training history)
        neg_item = np.random.randint(self.n_items)
        user_items = self.train_user_items.get(user, set())
        while neg_item in user_items:
            neg_item = np.random.randint(self.n_items)

        return int(user), int(pos_item), int(neg_item)


def get_train_loader(
    interaction_data: InteractionData,
    batch_size: int = BATCH_SIZE,
    seed: int = 42,
    num_workers: int | None = None,
    generator: "torch.Generator | None" = None,
) -> DataLoader:
    """Create training DataLoader with BPR sampling.

    The ``seed`` argument is forwarded to (a) a dedicated ``torch.Generator``
    that drives shuffling, and (b) a ``worker_init_fn`` that re-seeds NumPy
    and Python's ``random`` inside each DataLoader worker. Without these,
    ``num_workers > 0`` would use unseeded RNGs in each worker subprocess and
    negative-sample sequences would drift between runs even on identical
    hardware.

    On macOS, the default multiprocessing start method is ``spawn``, which
    cannot pickle the local ``_worker_init`` closure (raises
    ``AttributeError: Can't get local object``). Auto-fall-back to
    ``num_workers=0`` on Darwin keeps the loader usable on a Mac without
    affecting Linux servers (which use ``fork`` and pickle the closure fine).
    """
    import platform
    import random as _random

    def _worker_init(worker_id: int):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        _random.seed(worker_seed)

    if num_workers is None:
        num_workers = 0 if platform.system() == "Darwin" else 4

    # A caller (train.py) may supply its own generator so its state can be
    # checkpointed and restored on resume. Behaviour is identical when None.
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    dataset = BPRDataset(interaction_data)

    # `drop_last=True` discards a final partial batch so every BPR gradient is
    # estimated from the same number of pairs. On ML-20M (~9.9M training pairs)
    # that costs at most one batch. On a dataset smaller than one batch it costs
    # ALL of them: the loader yields nothing, the epoch loop never runs, and the
    # failure used to surface far away as `ZeroDivisionError: division by zero`
    # from `total_loss / n_batches` in train.py -- after the adjacency matrix had
    # been built and an epoch announced, so the traceback pointed at arithmetic
    # rather than at the setting that caused it. Say it here instead, where both
    # numbers are in hand.
    if len(dataset) < batch_size:
        raise ValueError(
            f"training set has {len(dataset):,} interaction pairs but batch_size is "
            f"{batch_size:,}, and the loader drops the final partial batch, so it "
            f"would yield 0 batches and training could not run.\n"
            f"Pass --batch-size N with N <= {len(dataset):,} (a power of two such as "
            f"{max(1, 1 << (max(1, len(dataset)).bit_length() - 1)):,} is a reasonable "
            f"choice here), or point the run at a larger dataset.\n"
            f"The published runs use the default batch_size={BATCH_SIZE:,} on datasets "
            f"of millions of pairs; changing it changes the optimisation, so a run with "
            f"a different batch size does not reproduce the paper's numbers."
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        generator=generator,
        worker_init_fn=_worker_init if num_workers > 0 else None,
    )
