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
from config import EMBEDDING_DIR, DATA_DIR


class ForeignFeatureFile(SystemExit):
    """A feature file whose row count this directory's movie_id_index.json cannot name.

    Its rows are indexed by another catalogue (a file copied in from another
    embedding directory, say), so aligning it here would hand items other items'
    vectors. A SystemExit subclass on purpose: uncaught, it ends a
    run with its message and no traceback; a caller that wants to record the refusal
    and carry on catches it BY NAME, before any generic handler. The attributes say
    what was refused.
    """

    def __init__(self, feature: str, path, rows: int, index_path, index_rows: int):
        self.feature, self.path, self.rows = feature, path, rows
        self.index_path, self.index_rows = index_path, index_rows
        super().__init__(
            f"feature {feature!r}: {path} has {rows:,} rows, and {index_path} names "
            f"{index_rows:,} -- the file belongs to another catalogue.\n"
            f"Build this feature for this catalogue, or point it at a file beside the index.")


class FeatureLoader:
    """
    Loads pre-computed item features and aligns them with benchmark item IDs.

    Usage:
        loader = FeatureLoader()
        profile_emb = loader.get_features("profile")    # (n_items, 1024) with the default encoder
        mood = loader.get_features("mood")               # (n_items, 10)
        combined = loader.get_combined(["profile", "mood"])  # (n_items, 1034) with the default encoder
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
            "bert_title": self.embedding_dir / "bert_title_embeddings.npy",
        }

        # Load embedding movie ID index (genome order).
        # The paper prints `EMBEDDING_DIR=output/my-encoder python3 run_experiment.py`
        # as the way to evaluate a new encoder without writing code, so this is where
        # a reader following that recipe arrives: say what the directory must contain.
        movie_id_index_path = self.embedding_dir / "movie_id_index.json"
        if not movie_id_index_path.exists():
            need = "\n".join(f"    {n}" for n in
                              ("movie_id_index.json", "profile_embeddings.npy",
                               "mood_vectors.npy", "theme_matrix.npy"))
            raise SystemExit(
                f"No feature index at {movie_id_index_path}.\n\n"
                f"EMBEDDING_DIR is {self.embedding_dir}, and a feature directory must\n"
                f"hold at least:\n{need}\n\n"
                "Fetch the ones we released:  python3 scripts/download_artifacts.py\n"
                "or encode your own:          python3 src/embedding_generator/main.py\n"
                "Vectors must be in the catalogue's index order, which movie_id_index.json\n"
                "declares -- see the extension section of the paper.")
        with open(movie_id_index_path) as f:
            genome_ids = json.load(f)
        self.genome_id_to_idx = {mid: i for i, mid in enumerate(genome_ids)}
        self.n_index = len(genome_ids)      # rows a feature file must have to align here

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
            # genome_raw (M2b) is the raw 1,128-d genome tag-relevance matrix. It
            # is not a file: it is built straight from genome-scores.csv, with the
            # same vectorised fill as eval_checkpoints.genome_raw_features so the
            # training and evaluation paths cannot diverge.
            if name == "genome_raw":
                return self._genome_raw()
            raise ValueError(f"Unknown feature: {name}. Available: {list(self._feature_files.keys())}")

        # Two of these five are NOT in the released download, because they encode
        # MovieLens content rather than generated text and are not ours to
        # redistribute: a reader has to build them from their own ML-20M copy.
        # Reaching np.load without this check produced a bare FileNotFoundError
        # traceback out of numpy -- the same defect the movie_id_index guard above
        # was written to fix, for a sibling path in this same file. The configs
        # that need them (M2, M3, M9) are in the paper's results table, so this is
        # the failure a reader following "skip stages 1-2" actually hits.
        path = self._feature_files[name]
        if not path.exists():
            how = {
                "genome": "python3 src/embedding_generator/main.py\n"
                          "        (PCA of the MovieLens genome scores)",
                "bert_title": "cd src/benchmark && python3 features/bert_baseline.py\n"
                              "        (bge-large-en-v1.5 encodings of the MovieLens titles;\n"
                              "        for e5-large-v2: python3 scripts/generate_e5_embeddings.py)",
            }.get(name)
            if how:
                raise FileNotFoundError(
                    f"feature {name!r} needs a file that is not in the release:\n"
                    f"    {path}\n"
                    "It encodes MovieLens content, which we do not redistribute, so\n"
                    "scripts/download_artifacts.py does not fetch it. Build it from your\n"
                    "own ML-20M copy:\n"
                    f"        {how}\n"
                    "Configurations needing it: genome -> M2, M9; bert_title -> M3.\n"
                    "The other configurations need nothing further.")
            raise FileNotFoundError(
                f"feature {name!r} is missing its file:\n    {path}\n"
                "Fetch the released embeddings with:\n"
                "        python3 scripts/download_artifacts.py")

        raw = np.load(path)
        # A file indexed by another catalogue cannot be aligned through THIS index:
        # it would hand items other items' rows, silently. Refuse a row count this
        # index cannot name.
        if raw.shape[0] != self.n_index:
            raise ForeignFeatureFile(name, path, raw.shape[0],
                                     self.embedding_dir / "movie_id_index.json", self.n_index)
        feat_dim = raw.shape[1]

        # Align to benchmark item IDs
        aligned = np.zeros((self.n_items, feat_dim), dtype=np.float32)
        for bench_id, genome_idx in self.alignment.items():
            aligned[bench_id] = raw[genome_idx]

        self._cache[name] = aligned
        return aligned

    def _genome_raw(self) -> np.ndarray:
        """Raw 1,128-d genome tag-relevance matrix for M2b, aligned to benchmark ids.

        Already aligned to item_map, so it does NOT go through self.alignment --
        that mapping is for arrays stored in genome-index order.
        """
        if "genome_raw" in self._cache:
            return self._cache["genome_raw"]

        import pandas as pd
        from config import GENOME_SCORES_CSV, GENOME_TAGS_CSV

        if not GENOME_SCORES_CSV.exists():
            raise FileNotFoundError(
                "feature 'genome_raw' (configuration M2b) is built from the MovieLens\n"
                "genome scores, which we do not redistribute:\n"
                f"    {GENOME_SCORES_CSV}\n"
                "Fetch MovieLens first:\n"
                "        bash scripts/download_ml20m.sh")

        tags = pd.read_csv(GENOME_TAGS_CSV)
        tag_idx = {tid: i for i, tid in enumerate(sorted(tags["tagId"].unique()))}
        feat = np.zeros((self.n_items, len(tags)), dtype=np.float32)

        s = pd.read_csv(GENOME_SCORES_CSV)
        s["bench"] = s["movieId"].map(self.item_map)
        s["tcol"] = s["tagId"].map(tag_idx)
        s = s.dropna(subset=["bench"])
        feat[s["bench"].to_numpy(dtype=int),
             s["tcol"].to_numpy(dtype=int)] = s["relevance"].to_numpy(dtype=np.float32)

        self._cache["genome_raw"] = feat
        return feat

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
