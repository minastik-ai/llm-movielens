#!/usr/bin/env python3
"""Experiment A for §3.5: zero-shot controllable retrieval via mood-axis offset.

For 100 query items spanning genres, generate two retrieval lists:
  L1 = top-10 nearest neighbors in profile-embedding space (standard)
  L2 = top-10 nearest in profile space, with the QUERY's mood vector offset
       by +0.5 along a chosen axis (e.g., serious_playful), and using mood-
       similarity as a re-ranking signal blended with profile similarity

Measure:
  - Mean shift in MOVED-AXIS COORDINATE between L1 and L2 retrieved items
    (positive shift = retrievals successfully shifted in the requested direction)
  - Per-axis "controllability score" = mean coordinate shift / requested offset
    Score of 1.0 = perfect controllability; 0 = no shift; <0 = opposite direction
  - Per-axis "drift on other axes" = mean change on non-target axes
    (low drift = clean axis-specific control; high drift = confounded)

Output: code/benchmark/results/analysis/mood_analysis/exp_a_controllable_retrieval.json
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
from _paths import code  # dual-layout paths: dev tree or published release

REPO = Path(__file__).resolve().parent.parent
EMB_DIR = code("embedding_generator", "output", "bge-large-en-v1.5")
META = json.load(open(code("embedding_generator", "output", "bge-large-en-v1.5", "embedding_metadata.json")))

AXES = ["dark_light", "serious_playful", "slow_fast", "cerebral_visceral",
        "realistic_fantastical", "intimate_epic", "conventional_experimental",
        "emotional_detached", "nostalgic_contemporary", "predictable_subversive"]
N_QUERIES = 100
TOP_K = 10
MOOD_OFFSET = 0.5
PROFILE_WEIGHT = 0.7   # blend factor for profile similarity vs mood similarity
SEED = 42

OUT_DIR = code("benchmark", "results", "analysis", "mood_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def cosine_topk(query_vec, all_vecs, k, exclude_idx=None):
    qn = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    an = all_vecs / (np.linalg.norm(all_vecs, axis=1, keepdims=True) + 1e-8)
    sim = an @ qn
    if exclude_idx is not None:
        sim[exclude_idx] = -np.inf
    return np.argpartition(-sim, k)[:k]


def euclidean_topk(query_vec, all_vecs, k, exclude_idx=None):
    """For mood vectors with semantic axes, euclidean is more interpretable."""
    diff = all_vecs - query_vec
    dist = np.linalg.norm(diff, axis=1)
    if exclude_idx is not None:
        dist[exclude_idx] = np.inf
    return np.argpartition(dist, k)[:k]


def main():
    rng = np.random.default_rng(SEED)
    profile = np.load(EMB_DIR / "profile_embeddings.npy")  # (10381, 1024)
    mood    = np.load(EMB_DIR / "mood_vectors.npy")        # (10381, 10)
    print(f"Loaded profile {profile.shape}, mood {mood.shape}")

    # 100 query indices
    queries = rng.choice(len(profile), size=N_QUERIES, replace=False)

    per_axis_results = {}
    for ax_idx, ax_name in enumerate(AXES):
        print(f"\nAxis {ax_idx}: {ax_name}")
        target_shifts_axis    = []  # shift on target axis (should be > 0 if controlled)
        drift_per_axis        = []  # mean abs shift on NON-target axes (should be small)
        baseline_shifts_axis  = []  # baseline shift on target axis (no offset applied) — control

        for q in queries:
            q_profile = profile[q]
            q_mood    = mood[q]

            # L1: standard top-K by profile cosine similarity
            l1 = cosine_topk(q_profile, profile, TOP_K, exclude_idx=q)
            l1_mood_centroid = mood[l1].mean(axis=0)

            # L2: target query mood with +0.5 on this axis; rerank by blend of profile+mood similarity
            target_mood = q_mood.copy()
            target_mood[ax_idx] = np.clip(target_mood[ax_idx] + MOOD_OFFSET, -1.0, 1.0)

            # Score = profile cosine + mood proximity (negative euclidean) blend
            qpn = q_profile / (np.linalg.norm(q_profile) + 1e-8)
            pn  = profile / (np.linalg.norm(profile, axis=1, keepdims=True) + 1e-8)
            profile_sim = pn @ qpn
            mood_dist   = np.linalg.norm(mood - target_mood, axis=1)
            # normalize mood_dist to [0,1] then convert to similarity
            mood_sim = 1.0 - (mood_dist / mood_dist.max())
            score = PROFILE_WEIGHT * profile_sim + (1 - PROFILE_WEIGHT) * mood_sim
            score[q] = -np.inf
            l2 = np.argpartition(-score, TOP_K)[:TOP_K]
            l2_mood_centroid = mood[l2].mean(axis=0)

            # Measure shifts
            mood_shifts = l2_mood_centroid - l1_mood_centroid  # (10,)
            target_shifts_axis.append(mood_shifts[ax_idx])
            other_axes = np.array([i for i in range(10) if i != ax_idx])
            drift_per_axis.append(np.abs(mood_shifts[other_axes]).mean())

            # Baseline control: rerun L2 with NO mood offset (target_mood = q_mood)
            mood_dist_b = np.linalg.norm(mood - q_mood, axis=1)
            mood_sim_b  = 1.0 - (mood_dist_b / mood_dist_b.max())
            score_b     = PROFILE_WEIGHT * profile_sim + (1 - PROFILE_WEIGHT) * mood_sim_b
            score_b[q]  = -np.inf
            l2_baseline = np.argpartition(-score_b, TOP_K)[:TOP_K]
            l2b_mood_centroid = mood[l2_baseline].mean(axis=0)
            baseline_shifts_axis.append((l2b_mood_centroid - l1_mood_centroid)[ax_idx])

        per_axis_results[ax_name] = {
            "axis_idx": ax_idx,
            "n_queries": N_QUERIES,
            "requested_offset": MOOD_OFFSET,
            "mean_target_axis_shift": float(np.mean(target_shifts_axis)),
            "std_target_axis_shift": float(np.std(target_shifts_axis, ddof=1)),
            "mean_baseline_shift": float(np.mean(baseline_shifts_axis)),
            "controllability_score": float(np.mean(target_shifts_axis) / MOOD_OFFSET),
            "mean_off_axis_drift": float(np.mean(drift_per_axis)),
        }
        print(f"  target shift mean = {np.mean(target_shifts_axis):+.3f}  "
              f"(baseline = {np.mean(baseline_shifts_axis):+.3f})  "
              f"controllability = {np.mean(target_shifts_axis)/MOOD_OFFSET:.2%}  "
              f"off-axis drift = {np.mean(drift_per_axis):.3f}")

    # Summary
    avg_ctrl = np.mean([r["controllability_score"] for r in per_axis_results.values()])
    avg_drift = np.mean([r["mean_off_axis_drift"] for r in per_axis_results.values()])
    out = {
        "protocol": (f"Zero-shot controllable retrieval. {N_QUERIES} random query items, "
                     f"+{MOOD_OFFSET} mood offset on target axis, top-{TOP_K} retrieval, "
                     f"score = {PROFILE_WEIGHT:.0%}*profile_cos + {1-PROFILE_WEIGHT:.0%}*mood_proximity. "
                     f"Controllability = (retrieved-set mean axis-coord shift) / requested_offset. "
                     f"Drift = mean abs shift on the other 9 axes (lower = cleaner control)."),
        "per_axis": per_axis_results,
        "mean_controllability": float(avg_ctrl),
        "mean_off_axis_drift": float(avg_drift),
        "ranked_high_to_low": sorted(
            [(ax, per_axis_results[ax]["controllability_score"]) for ax in AXES],
            key=lambda x: -x[1]),
    }
    out_path = OUT_DIR / "exp_a_controllable_retrieval.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {out_path}")
    print(f"\nMean controllability across 10 axes: {avg_ctrl:.2%}")
    print(f"Mean off-axis drift: {avg_drift:.3f}")
    print(f"\nRanked by controllability:")
    for ax, c in out["ranked_high_to_low"]:
        print(f"  {ax:<28} {c:>6.2%}")


if __name__ == "__main__":
    main()
