"""
Feature loader: maps pre-computed .npy embeddings to contiguous item IDs
used in the benchmark.

The embedding_generator outputs are indexed by genome movie IDs (via movie_id_index.json).
The benchmark uses contiguous 0-indexed item IDs (via item_map.json).
This module bridges the two.
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMBEDDING_DIR, DATA_DIR, BERT_TITLE_EMB_NPY


class FeatureLoader:
    """
    Loads pre-computed item features and aligns them with benchmark item IDs.

    Usage:
        loader = FeatureLoader()
        profile_emb = loader.get_features("profile")    # (n_items, 128)
        mood = loader.get_features("mood")               # (n_items, 10)
        combined = loader.get_combined(["profile", "mood"])  # (n_items, 138)
    """

    def __init__(self, data_dir: Path = None, embedding_dir: Path = None):
        if data_dir is None:
            data_dir = DATA_DIR
        if embedding_dir is None:
            embedding_dir = EMBEDDING_DIR

        self.embedding_dir = Path(embedding_dir)

        # Build feature file map relative to the actual embedding_dir
        self._feature_files = {
            "profile": self.embedding_dir / "profile_embeddings.npy",
            "mood": self.embedding_dir / "mood_vectors.npy",
            "themes": self.embedding_dir / "theme_matrix.npy",
            "genome": self.embedding_dir / "genome_embeddings.npy",
            "bert_title": BERT_TITLE_EMB_NPY,
        }

        # Load embedding movie ID index (genome order)
        movie_id_index_path = self.embedding_dir / "movie_id_index.json"
        with open(movie_id_index_path) as f:
            genome_ids = json.load(f)
        self.genome_id_to_idx = {mid: i for i, mid in enumerate(genome_ids)}

        # Load benchmark item mapping (original movieId → contiguous id)
        with open(Path(data_dir) / "item_map.json") as f:
            self.item_map = {int(k): v for k, v in json.load(f).items()}

        self.n_items = len(self.item_map)

        # Build alignment: benchmark_item_id → genome_embedding_row
        self.alignment = {}
        for original_mid, contiguous_id in self.item_map.items():
            if original_mid in self.genome_id_to_idx:
                self.alignment[contiguous_id] = self.genome_id_to_idx[original_mid]

        self._cache = {}

    def has_feature(self, name: str) -> bool:
        """Check if a feature file exists on disk."""
        if name not in self._feature_files:
            return False
        return self._feature_files[name].exists()

    def get_features(self, name: str) -> np.ndarray:
        """
        Load a feature matrix aligned to benchmark item IDs.
        Returns shape (n_items, feature_dim). Items without features get zeros.
        """
        if name in self._cache:
            return self._cache[name]

        if name not in self._feature_files:
            raise ValueError(f"Unknown feature: {name}. Available: {list(self._feature_files.keys())}")

        raw = np.load(self._feature_files[name])
        feat_dim = raw.shape[1]

        # Align to benchmark item IDs
        aligned = np.zeros((self.n_items, feat_dim), dtype=np.float32)
        for bench_id, genome_idx in self.alignment.items():
            aligned[bench_id] = raw[genome_idx]

        self._cache[name] = aligned
        return aligned

    def get_combined(self, feature_names: List[str]) -> np.ndarray:
        """Concatenate multiple features along dim=1."""
        if not feature_names:
            return None
        parts = [self.get_features(name) for name in feature_names]
        return np.concatenate(parts, axis=1)

    def get_features_tensor(self, name: str, device: str = "cpu") -> torch.Tensor:
        """Get features as a PyTorch tensor."""
        return torch.from_numpy(self.get_features(name)).to(device)

    def get_combined_tensor(self, feature_names: List[str], device: str = "cpu") -> torch.Tensor:
        """Get concatenated features as a PyTorch tensor."""
        combined = self.get_combined(feature_names)
        if combined is None:
            return None
        return torch.from_numpy(combined).to(device)

    def get_feature_dim(self, feature_names: List[str]) -> int:
        """Get total dimension of concatenated features."""
        if not feature_names:
            return 0
        return sum(self.get_features(name).shape[1] for name in feature_names)
