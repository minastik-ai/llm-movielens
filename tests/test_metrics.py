"""Tests for evaluation metrics correctness."""

from pathlib import Path

import numpy as np
import pytest

def _compute_metrics():
    """Load compute_metrics from whichever layout this checkout has.

    evaluate.py lives at code/benchmark/ in the development tree and at
    src/benchmark/ in the release; this finds it in either.
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


class TestMRR:
    """MRR is the reciprocal rank of the first relevant item over the FULL ranking."""

    def test_hit_inside_top_k(self):
        compute_metrics = _compute_metrics()
        scores = np.arange(100, 0, -1, dtype=float)       # item i has rank i + 1
        m = compute_metrics(scores, {4}, set(), 100, [3, 10])
        assert m["MRR"] == pytest.approx(1 / 5)

    def test_hit_beyond_top_k_is_not_truncated(self):
        compute_metrics = _compute_metrics()
        scores = np.arange(200, 0, -1, dtype=float)       # item i has rank i + 1
        m = compute_metrics(scores, {119}, set(), 200, [10, 20, 50])
        assert m["MRR"] == pytest.approx(1 / 120)          # was 0.0 when cut at max(top_k)

    def test_first_of_several_relevant_items_counts(self):
        compute_metrics = _compute_metrics()
        scores = np.arange(200, 0, -1, dtype=float)
        m = compute_metrics(scores, {150, 99, 180}, set(), 200, [10, 50])
        assert m["MRR"] == pytest.approx(1 / 100)

    def test_training_items_are_pushed_to_the_bottom(self):
        compute_metrics = _compute_metrics()
        scores = np.arange(200, 0, -1, dtype=float)
        train = set(range(0, 60))                          # removes ranks 1..60
        m = compute_metrics(scores, {119}, train, 200, [10, 20, 50])
        assert m["MRR"] == pytest.approx(1 / 60)           # 119 now has rank 120 - 60

    def test_matches_a_full_sort_on_random_scores(self):
        """Both branches (hit inside and beyond the top K) agree with argsort."""
        compute_metrics = _compute_metrics()
        rng = np.random.default_rng(0)
        n_items = 500
        for _ in range(200):
            scores = rng.normal(size=n_items)
            gt = set(rng.choice(n_items, size=int(rng.integers(1, 6)), replace=False).tolist())
            train = set(rng.choice(n_items, size=30, replace=False).tolist()) - gt
            masked = scores.copy(); masked[list(train)] = -np.inf
            order = np.argsort(-masked)
            expected = 1.0 / (1 + min(int(np.where(order == i)[0][0]) for i in gt))
            got = compute_metrics(scores, gt, train, n_items, [10, 20, 50])["MRR"]
            assert got == pytest.approx(expected)

    def test_never_below_the_truncated_value(self):
        compute_metrics = _compute_metrics()
        rng = np.random.default_rng(1)
        scores = rng.normal(size=300)
        gt = {int(np.argsort(scores)[3])}                  # a deep hit
        m = compute_metrics(scores, gt, set(), 300, [10, 50])
        assert m["MRR"] > 0.0
