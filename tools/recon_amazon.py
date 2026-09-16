#!/usr/bin/env python3
"""Stage-0 recon for adding an Amazon Reviews 2023 category as a third dataset.

WHY THIS EXISTS. Adding a dataset to the benchmark costs LLM spend and GPU time,
and the two figures that set both are unknown until the protocol is actually
applied: how many users and items survive iterative k-core at Amazon's ~1.7
interactions per user, and how much metadata the surviving items carry for the
generator to reason over. This script answers both from the real files, without
training anything and without generating a single profile.

It applies EXACTLY the protocol in rebuild_splits.py -- same positive threshold,
same iterative k-core over users and items, same train-only id remapping -- so
the numbers it reports are comparable with the paper's ML-20M row rather than
merely suggestive.

    python3 recon_amazon.py --category Video_Games

Downloads (cached): the rating-only CSV (~264 MB for Video_Games) and the item
metadata JSONL (~437 MB). Requires pandas. No GPU, no API key, no network after
the first run.
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main"

# --- Protocol constants, copied from rebuild_splits.py. Do not diverge. -------
POSITIVE_THRESHOLD = 3.5
K_CORE = 10

# --- ML-20M anchors, for scaling. Measured, not estimated. -------------------
ML20M = {
    "users": 127_371, "items": 9_906, "train": 11_499_778,
    "val": 49_668, "test": 67_466,
    "gen_cost_usd": 21.0, "gen_profiles": 10_381, "gen_input_tokens": 2_012,
    # MEASURED, not assumed: 399.6 in generation_economics.json, and a live
    # 5-item batch on 2026-09-02 returned 395. The earlier 130 was ~3x too low
    # and, being inside the calibration, distorted every extrapolation away
    # from ML-20M's own input length.
    "gen_output_tokens": 400,           # 80-120 words, FIXED regardless of input
    "price_in_per_tok": 0.5e-6,         # Haiku 4.5, batch (50% off list $1/M)
    "price_out_per_tok": 2.5e-6,        # Haiku 4.5, batch (50% off list $5/M)
    "gpu_hours_per_run_cuda": 3.0,      # median over 32 CUDA-class runs
    "gpu_hours_per_run_mps": 10.6,      # median over 68 Apple-silicon runs
}
# The paper's split is ~98.99 / 0.43 / 0.58 percent of interactions.
SPLIT_FRACTIONS = (ML20M["train"], ML20M["val"], ML20M["test"])


# The system prompt is charged IN FULL on every call: at ~1,200-1,400 tokens it is
# below Claude Haiku 4.5's 4,096-token minimum cacheable prefix, so no cache entry
# is ever created. The ML-20M anchor (2,012) is TOTAL input per profile, so any
# candidate dataset must be costed on payload + system prompt, never payload alone.
def cost_per_profile(input_tokens: float) -> float:
    """Dollars per generated profile at a given prompt length.

    Do NOT scale the measured $/profile by input length alone. The output is
    pinned at 80-120 words whatever the input, so on a metadata-poor catalogue
    the OUTPUT dominates and a purely input-scaled estimate understates the bill
    several-fold -- it put Steam at $1 when the real figure is closer to $6.
    We price input and output separately and calibrate the pair so that ML-20M
    reproduces its measured total.
    """
    pi, po = ML20M["price_in_per_tok"], ML20M["price_out_per_tok"]
    out = ML20M["gen_output_tokens"]
    measured = ML20M["gen_cost_usd"] / ML20M["gen_profiles"]
    modelled = ML20M["gen_input_tokens"] * pi + out * po
    calib = measured / modelled          # absorbs retries and cache misses
    return calib * (input_tokens * pi + out * po)


def log(m):
    print(m, flush=True)


def fetch(url: str, dest: Path) -> Path:
    """Download once, then reuse. curl -C - resumes a partial file."""
    if dest.exists() and dest.stat().st_size > 0:
        log(f"  cached  {dest.name} ({dest.stat().st_size/1e6:.0f} MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"  fetching {dest.name} ...")
    t0 = time.time()
    r = subprocess.run(["curl", "-sSL", "-C", "-", "-o", str(dest), url])
    if r.returncode != 0 or not dest.exists():
        sys.exit(f"ERROR: download failed for {url}")
    log(f"  got      {dest.name} ({dest.stat().st_size/1e6:.0f} MB in {time.time()-t0:.0f}s)")
    return dest


def k_core_filter(df, k=K_CORE):
    """Iteratively drop users and items with fewer than k interactions.

    Identical to rebuild_splits.py: iterative because removing a sparse user can
    push an item below threshold and vice versa.
    """
    it = 0
    trace = []
    while True:
        it += 1
        n0 = len(df)
        uc = df["user_id"].value_counts()
        df = df[df["user_id"].isin(uc[uc >= k].index)]
        ic = df["parent_asin"].value_counts()
        df = df[df["parent_asin"].isin(ic[ic >= k].index)]
        trace.append((it, len(df), df["user_id"].nunique(), df["parent_asin"].nunique()))
        if len(df) == n0:
            break
        if it > 50:
            log("  WARNING: k-core did not converge in 50 iterations")
            break
    return df, trace


def temporal_split(df):
    """Cut at the timestamps that reproduce the paper's train/val/test shares."""
    # sorted() rather than in-place: to_numpy() may hand back a read-only view.
    ts = sorted(df["timestamp"].tolist())
    tot = SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1] + SPLIT_FRACTIONS[2]
    i_tr = int(len(ts) * SPLIT_FRACTIONS[0] / tot)
    i_va = int(len(ts) * (SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1]) / tot)
    return int(ts[i_tr]), int(ts[i_va])


def scan_metadata(path: Path, keep: set):
    """Stream the metadata, keeping only surviving items. 437 MB is not loaded."""
    fields = ["title", "description", "features", "categories", "details",
              "store", "price", "average_rating", "rating_number"]
    present = {f: 0 for f in fields}
    payload_chars, cat_depths, seen = [], [], set()
    n_lines = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            n_lines += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            a = r.get("parent_asin")
            if a not in keep or a in seen:
                continue
            seen.add(a)
            for f in fields:
                v = r.get(f)
                if v not in (None, "", [], {}):
                    present[f] += 1
            # What the generator would actually be shown, mirroring the ML-20M prompt:
            # the structured description + the free text + the aggregates.
            parts = [str(r.get("title") or "")]
            parts += [str(x) for x in (r.get("description") or [])]
            parts += [str(x) for x in (r.get("features") or [])]
            parts += [" > ".join(str(x) for x in (r.get("categories") or []))]
            parts += [f"{k}: {v}" for k, v in (r.get("details") or {}).items()]
            payload_chars.append(len(" ".join(parts)))
            cat_depths.append(len(r.get("categories") or []))
    return present, payload_chars, cat_depths, len(seen), n_lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", default="Video_Games")
    ap.add_argument("--k-core", type=int, default=K_CORE)
    ap.add_argument("--cache", type=Path, default=Path("/tmp/amazon_recon"))
    ap.add_argument("--limit", type=int, default=0,
                    help="read only N rating rows — for testing the logic, not for real numbers")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    cat = a.category
    log(f"\nStage-0 recon — Amazon Reviews 2023 / {cat}"
        f"{'  [LIMITED SAMPLE, NOT VALID NUMBERS]' if a.limit else ''}\n")

    ratings = fetch(f"{REPO}/benchmark/0core/rating_only/{cat}.csv",
                    a.cache / f"{cat}.csv")
    meta = fetch(f"{REPO}/raw/meta_categories/meta_{cat}.jsonl",
                 a.cache / f"meta_{cat}.jsonl")

    log("\n--- interactions ---")
    df = pd.read_csv(ratings, nrows=a.limit or None)
    raw_n, raw_u, raw_i = len(df), df["user_id"].nunique(), df["parent_asin"].nunique()
    log(f"  raw                       {raw_n:>12,} int  {raw_u:>10,} users  {raw_i:>8,} items"
        f"   ({raw_n/raw_u:.2f} int/user, {raw_n/raw_i:.1f} int/item)")

    df = df[df["rating"] >= POSITIVE_THRESHOLD].drop(columns=["rating"])
    log(f"  after rating >= {POSITIVE_THRESHOLD}       {len(df):>12,} int"
        f"  {df['user_id'].nunique():>10,} users  {df['parent_asin'].nunique():>8,} items")

    df, trace = k_core_filter(df, a.k_core)
    shown = trace if len(trace) <= 5 else trace[:4] + [None] + trace[-1:]
    for row in shown:
        if row is None:
            log("      …")
            continue
        it, n, u, i = row
        log(f"    iter {it:<2}                  {n:>12,} int  {u:>10,} users  {i:>8,} items")
    if df.empty:
        sys.exit(f"\n  STOP: {a.k_core}-core leaves nothing. The protocol does not "
                 f"transfer to this category at k={a.k_core}.")

    n, nu, ni = len(df), df["user_id"].nunique(), df["parent_asin"].nunique()
    dens = n / (nu * ni)
    log(f"\n  after {a.k_core}-core            {n:>12,} int  {nu:>10,} users  {ni:>8,} items")
    log(f"  retained                  {100*n/raw_n:>11.1f}% int  {100*nu/raw_u:>9.1f}% users  {100*ni/raw_i:>7.1f}% items")
    log(f"  density {dens*100:.3f}%   {n/ni:.1f} int/item   {n/nu:.1f} int/user"
        f"      [ML-20M: 0.910%, {ML20M['train']/ML20M['items']:.0f} int/item]")

    # Guard the same unit bug that bit the Steam loader: these are native ms.
    yr = lambda ms: int(time.strftime("%Y", time.gmtime(ms / 1000)))
    if not (1995 <= yr(df["timestamp"].min()) and yr(df["timestamp"].max()) <= 2030):
        sys.exit("ERROR: implausible date range — timestamp unit bug")
    t_tr, t_va = temporal_split(df)
    tr = int((df["timestamp"] < t_tr).sum())
    va = int(((df["timestamp"] >= t_tr) & (df["timestamp"] < t_va)).sum())
    te = int((df["timestamp"] >= t_va).sum())
    fmt = lambda ms: time.strftime("%Y-%m-%d", time.gmtime(ms / 1000))
    log(f"\n--- temporal split (cut to match the paper's 99.0/0.4/0.6 shares) ---")
    log(f"  train < {fmt(t_tr)}        {tr:>12,}")
    log(f"  val   < {fmt(t_va)}        {va:>12,}")
    log(f"  test                       {te:>12,}")
    if va < 5000 or te < 5000:
        log("  WARNING: val/test windows are small; consider a different cut.")

    log("\n--- item metadata for the surviving items ---")
    keep = set(df["parent_asin"].unique())
    present, chars, depths, matched, scanned = scan_metadata(meta, keep)
    log(f"  matched {matched:,} of {ni:,} surviving items in {scanned:,} metadata records")
    if matched == 0:
        sys.exit("  STOP: no metadata matched. Check the id field.")
    for f, c in present.items():
        log(f"    {f:<16}{100*c/matched:>6.1f}% present")
    med_chars = statistics.median(chars)
    est_tok = med_chars / 3.8   # ~3.8 chars/token for English prose
    log(f"  prompt payload: median {med_chars:,.0f} chars -> ~{est_tok:,.0f} tokens"
        f"   [ML-20M: {ML20M['gen_input_tokens']:,} tokens]")
    log(f"  category depth: median {statistics.median(depths):.0f} levels")

    log("\n--- projected cost ---")
    per_profile = cost_per_profile(est_tok)
    gen = ni * per_profile
    log(f"  generation   {ni:,} profiles x ${per_profile:.5f} = ${gen:,.2f}"
        f"   (calibrated so ML-20M reproduces ${ML20M['gen_cost_usd']:.0f})")
    f_int = n / ML20M["train"]
    f_eval = (nu * ni) / (ML20M["users"] * ML20M["items"])
    lo, hi = sorted((f_int, f_eval))
    log(f"  per-epoch scaling vs ML-20M: interactions {f_int:.2f}x, users x items {f_eval:.2f}x")
    for label, h in (("CUDA-class", ML20M["gpu_hours_per_run_cuda"]),
                     ("Apple-silicon", ML20M["gpu_hours_per_run_mps"])):
        log(f"  20 runs (4 cfg x 5 seeds), {label:<14} {20*h*lo:>6.0f} - {20*h*hi:>5.0f} GPU-hours")
    log("  (bounds assume per-epoch cost is dominated by one term or the other;")
    log("   Stage 1 measures the true mix on one seed.)")

    out = a.out or Path(f"recon_{cat}.json")
    json.dump({
        "category": cat, "k_core": a.k_core, "limited_sample": bool(a.limit),
        "raw": {"interactions": raw_n, "users": raw_u, "items": raw_i},
        "post_kcore": {"interactions": n, "users": nu, "items": ni, "density": dens,
                       "int_per_item": n / ni, "int_per_user": n / nu},
        "split": {"train": tr, "val": va, "test": te,
                  "cut_train_ms": t_tr, "cut_val_ms": t_va},
        "metadata": {"matched": matched,
                     "coverage_pct": {k: 100 * v / matched for k, v in present.items()},
                     "median_payload_chars": med_chars, "est_input_tokens": est_tok},
        "projected": {"generation_usd": gen,
                      "gpu_hours_cuda": [20 * ML20M["gpu_hours_per_run_cuda"] * lo,
                                         20 * ML20M["gpu_hours_per_run_cuda"] * hi]},
    }, open(out, "w"), indent=2)
    log(f"\n  wrote {out}\n")


if __name__ == "__main__":
    main()
