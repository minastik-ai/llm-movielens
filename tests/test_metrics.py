"""Tests for evaluation metrics correctness."""

from pathlib import Path

import numpy as np
import pytest

def _compute_metrics():
    """Load compute_metrics from whichever layout this checkout has.

    The eight tests below imported `llm_movielens.benchmark.evaluate` -- a package
    that exists in NEITHER the development tree nor the release, so every one of
    them had been failing with ModuleNotFoundError on a phantom namespace.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _layout import load_module
    return load_module("benchmark", "evaluate.py").compute_metrics



class TestNDCG:
    """Verify NDCG computation on known examples."""

    def test_perfect_ranking(self):
        """NDCG@K should be 1.0 for a perfect ranking."""
        # Simulate: top-K items are exactly the ground truth
        compute_metrics = _compute_metrics()

        n_items = 100
        scores = np.zeros(n_items)
        ground_truth = {0, 1, 2}  # items 0, 1, 2 are relevant
        # Give highest scores to relevant items
        scores[0] = 10.0
        scores[1] = 9.0
        scores[2] = 8.0

        metrics = compute_metrics(scores, ground_truth, set(), n_items, [3, 10])
        assert metrics["NDCG@3"] == pytest.approx(1.0, abs=1e-6)

    def test_worst_ranking(self):
        """NDCG@K should be 0.0 when no relevant items are in top-K."""
        compute_metrics = _compute_metrics()

        n_items = 100
        scores = np.zeros(n_items)
        ground_truth = {90, 91, 92}  # relevant items at the end
        # Give highest scores to irrelevant items
        scores[0] = 10.0
        scores[1] = 9.0
        scores[2] = 8.0

        metrics = compute_metrics(scores, ground_truth, set(), n_items, [3])
        assert metrics["NDCG@3"] == pytest.approx(0.0, abs=1e-6)

    def test_ndcg_monotonic(self):
        """NDCG@K should be non-decreasing as K increases."""
        compute_metrics = _compute_metrics()

        n_items = 100
        np.random.seed(42)
        scores = np.random.rand(n_items)
        ground_truth = {10, 20, 30, 40, 50}

        metrics = compute_metrics(scores, ground_truth, set(), n_items, [10, 20, 50])
        assert metrics["NDCG@10"] <= metrics["NDCG@20"] + 1e-6
        assert metrics["NDCG@20"] <= metrics["NDCG@50"] + 1e-6


class TestRecall:
    """Verify Recall computation."""

    def test_perfect_recall(self):
        compute_metrics = _compute_metrics()

        n_items = 100
        scores = np.zeros(n_items)
        ground_truth = {0, 1}
        scores[0] = 10.0
        scores[1] = 9.0

        metrics = compute_metrics(scores, ground_truth, set(), n_items, [10])
        assert metrics["Recall@10"] == pytest.approx(1.0, abs=1e-6)

    def test_partial_recall(self):
        compute_metrics = _compute_metrics()

        n_items = 100
        scores = np.zeros(n_items)
        ground_truth = {0, 1, 50, 60}
        scores[0] = 10.0  # relevant, in top-2
        scores[1] = 9.0   # relevant, in top-2
        scores[50] = 1.0  # relevant, but lower ranked
        scores[60] = 0.5  # relevant, but lower ranked

        metrics = compute_metrics(scores, ground_truth, set(), n_items, [2])
        assert metrics["Recall@2"] == pytest.approx(0.5, abs=1e-6)  # 2 out of 4


class TestHitRate:
    """Verify Hit Rate computation."""

    def test_hit(self):
        compute_metrics = _compute_metrics()

        n_items = 100
        scores = np.zeros(n_items)
        ground_truth = {0}
        scores[0] = 10.0

        metrics = compute_metrics(scores, ground_truth, set(), n_items, [10])
        assert metrics["HR@10"] == pytest.approx(1.0, abs=1e-6)

    def test_miss(self):
        compute_metrics = _compute_metrics()

        n_items = 100
        scores = np.zeros(n_items)
        ground_truth = {99}
        scores[0] = 10.0  # irrelevant item ranked first

        metrics = compute_metrics(scores, ground_truth, set(), n_items, [1])
        assert metrics["HR@1"] == pytest.approx(0.0, abs=1e-6)


class TestTrainItemExclusion:
    """Verify that training items are excluded from ranking."""

    def test_train_items_excluded(self):
        compute_metrics = _compute_metrics()

        n_items = 100
        scores = np.zeros(n_items)
        train_items = {0, 1, 2}  # these should be masked
        ground_truth = {3}
        scores[0] = 100.0  # highest score but in train set
        scores[3] = 5.0    # relevant test item

        metrics = compute_metrics(scores, ground_truth, train_items, n_items, [1])
        # Item 0 should be masked, so item 3 should be ranked first
        assert metrics["HR@1"] == pytest.approx(1.0, abs=1e-6)
