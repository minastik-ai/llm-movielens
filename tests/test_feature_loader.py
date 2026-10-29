"""Tests for feature loading and mapping."""

import json
from pathlib import Path

import numpy as np
import pytest

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _layout import component as _component
# The generator's top-level output/ holds a PCA-128 variant under the SAME
# filenames as the 1024-d arrays one directory down, so point at the encoder.
_gen = _component("embedding_generator")
EMB = ((_gen / "output" / "bge-large-en-v1.5") if _gen else None) or Path(".")


class TestMovieIdIndex:
    """Verify movie ID index consistency."""

    @pytest.fixture
    def index_path(self):
        candidates = [
            EMB / "movie_id_index.json",
            EMB / "movie_id_index.json",
        ]
        for p in candidates:
            if p.exists():
                return p
        pytest.skip("movie_id_index.json not found")

    def test_index_count(self, index_path):
        with open(index_path) as f:
            index = json.load(f)
        assert len(index) == 10381

    def test_index_unique(self, index_path):
        with open(index_path) as f:
            index = json.load(f)
        ids = [entry["movieId"] if isinstance(entry, dict) else entry for entry in index]
        assert len(set(ids)) == len(ids), "Duplicate movieIds in index"


class TestThemeVocabulary:
    """Verify theme vocabulary consistency."""

    @pytest.fixture
    def vocab_path(self):
        candidates = [
            EMB / "theme_vocabulary.json",
            EMB / "theme_vocabulary.json",
        ]
        for p in candidates:
            if p.exists():
                return p
        pytest.skip("theme_vocabulary.json not found")

    def test_vocab_count(self, vocab_path):
        with open(vocab_path) as f:
            vocab = json.load(f)
        assert len(vocab) == 528, f"Expected 528 themes, got {len(vocab)}"

    def test_vocab_unique(self, vocab_path):
        with open(vocab_path) as f:
            vocab = json.load(f)
        assert len(set(vocab)) == len(vocab), "Duplicate themes in vocabulary"

    def test_theme_matrix_matches_vocab(self, vocab_path):
        """Theme matrix columns should match vocabulary size."""
        theme_candidates = [
            EMB / "theme_matrix.npy",
            EMB / "theme_matrix.npy",
        ]
        theme_path = None
        for p in theme_candidates:
            if p.exists():
                theme_path = p
                break
        if theme_path is None:
            pytest.skip("theme_matrix.npy not found")

        with open(vocab_path) as f:
            vocab = json.load(f)
        matrix = np.load(theme_path)
        assert matrix.shape[1] == len(vocab)
