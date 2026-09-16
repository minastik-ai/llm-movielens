#!/usr/bin/env python3
"""Stage-0 recon for adding the Steam dataset (McAuley) as a third dataset.

Companion to recon_amazon.py, which it imports the tested k-core and temporal-split
from, so both candidates are judged by identical code. Steam differs from Amazon in
three ways that this script has to handle and that the paper has to disclose:

  1. The files are PYTHON REPR, not JSON (u'key': u'value'), so json.loads fails.
     Parsing 7.8M lines with ast.literal_eval is slow, so we extract the three
     fields we need by regex and VERIFY the fast path against literal_eval on a
     sample before trusting it.
  2. There are NO STAR RATINGS. The paper's "rating >= 3.5" rule cannot transfer.
     A Steam review is itself the positive signal, optionally gated on hours played.
  3. The ownership file (australian_users_items) has NO TIMESTAMPS at all, so the
     paper's temporal split is impossible from it. Only steam_reviews.json.gz
     carries a date, which is why this script uses the reviews as interactions.

    python3 recon_steam.py

Downloads (cached): steam_reviews.json.gz (~1.3 GB) and steam_games.json.gz (~3 MB).
No GPU, no API key.
"""
import argparse
import ast
import gzip
import json
import re
import statistics
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recon_amazon import (ML20M, cost_per_profile, fetch, k_core_filter,  # noqa: E402
                          temporal_split)

# Canonical source is McAuley's page (cite Kang & McAuley ICDM 2018 / Wan &
# McAuley RecSys 2018). We fetch from the HuggingFace mirror because the UCSD
# host serves at ~195 KB/s and drops the connection; HF serves the identical
# file at ~3.8 MB/s. Verify with the size check below if you swap hosts.
REVIEWS_URL = ("https://huggingface.co/datasets/recommender-system/"
               "steam-review-and-bundle-dataset/resolve/main/steam_reviews.json.gz")
GAMES_URL = "https://cseweb.ucsd.edu/~wckang/steam_games.json.gz"
REVIEWS_ORIGIN = "https://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz"
REVIEWS_BYTES = 1_338_063_248

# Fast field extraction. product_id and date are simple tokens; username may hold
# quotes, so it is the one we most need the literal_eval cross-check for.
RX = {k: re.compile(r"u?'%s':\s*u?'((?:[^'\\]|\\.)*)'" % k) for k in
      ("username", "product_id", "date")}
RX_HOURS = re.compile(r"u?'hours':\s*([0-9.]+)")


def log(m):
    print(m, flush=True)


def parse_fast(line):
    out = {}
    for k, rx in RX.items():
        m = rx.search(line)
        if not m:
            return None
        out[k] = m.group(1)
    h = RX_HOURS.search(line)
    out["hours"] = float(h.group(1)) if h else 0.0
    return out


def verify_fast_path(path, n=20000):
    """Cross-check the regex against ast.literal_eval before trusting it.

    product_id and date must match EXACTLY -- they are ASCII and they carry the
    interaction. The username need not: the regex leaves non-ASCII names in their
    escaped form (\\u20ae...) where literal_eval decodes them. That is harmless
    because the username is only an identity key, but only if the escaping is a
    BIJECTION -- so we verify that no two distinct real usernames collapse onto
    the same key, rather than assuming it. Collisions would silently merge users
    and inflate the k-core.
    """
    exact = bad = skipped = 0
    fwd, rev = {}, {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= n:
                break
            try:
                truth = ast.literal_eval(line)
            except Exception:
                skipped += 1
                continue
            got = parse_fast(line)
            if got is None:
                skipped += 1
                continue
            if (str(truth.get("product_id")) != got["product_id"]
                    or str(truth.get("date")) != got["date"]):
                bad += 1
                continue
            t, g = str(truth.get("username")), got["username"]
            exact += (t == g)
            if fwd.setdefault(t, g) != g or rev.setdefault(g, t) != t:
                bad += 1
    log(f"  fast-path check on {n:,} lines: item/date exact on all but {bad}, "
        f"{exact:,} usernames byte-identical, {len(fwd):,} distinct users bijective, "
        f"{skipped} unparsed")
    if bad:
        sys.exit(f"ERROR: fast path disagrees on {bad} lines — do not trust it")
    return exact


def load_reviews(path, min_hours):
    rows, dropped = [], 0
    t0 = time.time()
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh, 1):
            r = parse_fast(line)
            if r is None:
                dropped += 1
                continue
            if min_hours and r["hours"] < min_hours:
                continue
            rows.append((r["username"], r["product_id"], r["date"]))
            if i % 2_000_000 == 0:
                log(f"    {i:,} lines ({time.time()-t0:.0f}s)")
    log(f"  parsed {len(rows):,} interactions, {dropped:,} unparsable ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows, columns=["user_id", "parent_asin", "date"])
    # Explicit epoch-milliseconds. Do NOT use .astype("int64") // 10**6: pandas 2
    # picks its own datetime resolution (s / ms / us / ns) per column, so that
    # expression silently yields a different unit on different pandas versions --
    # it produced seconds here and dated every review to 1970.
    ts = pd.to_datetime(df["date"], errors="coerce", format="%Y-%m-%d")
    df["timestamp"] = ((ts - pd.Timestamp("1970-01-01"))
                       .dt.total_seconds().astype("float") * 1000)
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].astype("int64")
    df = df[df["timestamp"] > 0].drop(columns=["date"])
    lo, hi = df["timestamp"].min(), df["timestamp"].max()
    yr = lambda ms: time.strftime("%Y", time.gmtime(ms / 1000))
    log(f"  date range {yr(lo)}-{yr(hi)}")
    if not (1995 <= int(yr(lo)) and int(yr(hi)) <= 2030):
        sys.exit(f"ERROR: implausible date range {yr(lo)}-{yr(hi)} — unit bug")
    return df


def scan_games(path, keep):
    fields = ["title", "app_name", "genres", "tags", "specs", "developer",
              "publisher", "price", "release_date", "sentiment"]
    present = {f: 0 for f in fields}
    payload, ntags, seen = [], [], set()
    scanned = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            scanned += 1
            try:
                r = ast.literal_eval(line)
            except Exception:
                continue
            i = str(r.get("id"))
            if i not in keep or i in seen:
                continue
            seen.add(i)
            for f in fields:
                v = r.get(f)
                if v not in (None, "", [], {}):
                    present[f] += 1
            parts = [str(r.get("title") or r.get("app_name") or "")]
            parts += [", ".join(r.get("genres") or [])]
            parts += [", ".join(r.get("tags") or [])]
            parts += [", ".join(r.get("specs") or [])]
            parts += [str(r.get("developer") or ""), str(r.get("publisher") or "")]
            payload.append(len(" ".join(parts)))
            ntags.append(len(r.get("tags") or []))
    return present, payload, ntags, len(seen), scanned


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k-core", type=int, default=10)
    ap.add_argument("--min-hours", type=float, default=0.0,
                    help="positivity gate on hours played; 0 = any review counts")
    ap.add_argument("--cache", type=Path, default=Path("/tmp/steam_recon"))
    ap.add_argument("--out", type=Path, default=Path("recon_Steam.json"))
    a = ap.parse_args()

    log("\nStage-0 recon — Steam (McAuley)\n")
    rev = fetch(REVIEWS_URL, a.cache / "steam_reviews.json.gz")
    games = fetch(GAMES_URL, a.cache / "steam_games.json.gz")

    log("\n--- parsing (python-repr, not JSON) ---")
    verify_fast_path(rev)
    df = load_reviews(rev, a.min_hours)

    log("\n--- interactions ---")
    log("  NOTE: Steam has no star ratings. A review IS the positive signal"
        f"{f' (hours >= {a.min_hours})' if a.min_hours else ''}; the paper's"
        " rating>=3.5 rule does not transfer.")
    raw_n, raw_u, raw_i = len(df), df["user_id"].nunique(), df["parent_asin"].nunique()
    log(f"  raw                       {raw_n:>12,} int  {raw_u:>10,} users  {raw_i:>8,} items"
        f"   ({raw_n/raw_u:.2f} int/user, {raw_n/raw_i:.1f} int/item)")

    df, trace = k_core_filter(df, a.k_core)
    shown = trace if len(trace) <= 5 else trace[:4] + [None] + trace[-1:]
    for row in shown:
        if row is None:
            log("      …")
            continue
        it, n, u, i = row
        log(f"    iter {it:<2}                  {n:>12,} int  {u:>10,} users  {i:>8,} items")
    if df.empty:
        sys.exit(f"\n  STOP: {a.k_core}-core leaves nothing.")

    n, nu, ni = len(df), df["user_id"].nunique(), df["parent_asin"].nunique()
    dens = n / (nu * ni)
    log(f"\n  after {a.k_core}-core            {n:>12,} int  {nu:>10,} users  {ni:>8,} items")
    log(f"  retained                  {100*n/raw_n:>11.1f}% int  {100*nu/raw_u:>9.1f}% users  {100*ni/raw_i:>7.1f}% items")
    log(f"  density {dens*100:.3f}%   {n/ni:.1f} int/item   {n/nu:.1f} int/user"
        f"      [ML-20M: 0.910%, {ML20M['train']/ML20M['items']:.0f} int/item]")

    t_tr, t_va = temporal_split(df)
    tr = int((df["timestamp"] < t_tr).sum())
    va = int(((df["timestamp"] >= t_tr) & (df["timestamp"] < t_va)).sum())
    te = int((df["timestamp"] >= t_va).sum())
    fmt = lambda ms: time.strftime("%Y-%m-%d", time.gmtime(ms / 1000))
    log(f"\n--- temporal split (paper's 99.0/0.4/0.6 shares) ---")
    log(f"  train < {fmt(t_tr)}        {tr:>12,}")
    log(f"  val   < {fmt(t_va)}        {va:>12,}")
    log(f"  test                       {te:>12,}")
    if va < 5000 or te < 5000:
        log("  WARNING: val/test windows are small; consider a different cut.")

    log("\n--- item metadata for the surviving items ---")
    present, chars, ntags, matched, scanned = scan_games(games, set(df["parent_asin"].unique()))
    log(f"  matched {matched:,} of {ni:,} surviving items in {scanned:,} game records")
    if matched == 0:
        sys.exit("  STOP: no metadata matched — check the id field.")
    for f, c in present.items():
        log(f"    {f:<16}{100*c/matched:>6.1f}% present")
    med = statistics.median(chars)
    est_tok = med / 3.8
    log(f"  prompt payload: median {med:,.0f} chars -> ~{est_tok:,.0f} tokens"
        f"   [ML-20M: {ML20M['gen_input_tokens']:,}; Amazon VG: ~509]")
    log(f"  tags per game: median {statistics.median(ntags):.0f}")
    log("  NOTE: Steam ships no free-text description. The payload is a tag/genre")
    log("        list, so the generator has far less to reason over than on ML-20M.")

    log("\n--- projected cost ---")
    # payload + system prompt: the system prompt is not cacheable and is billed
    # on every call, so costing the payload alone understates by ~3x here.
    SYSTEM_PROMPT_TOKENS = 1_210      # settings_steam.py, 4,599 chars / 3.8
    per = cost_per_profile(est_tok + SYSTEM_PROMPT_TOKENS)
    log(f"  generation   {ni:,} profiles x ${per:.5f} = ${ni*per:,.2f}"
        f"   (output is fixed at ~{ML20M['gen_output_tokens']} tokens and dominates here)")
    f_int = n / ML20M["train"]
    f_eval = (nu * ni) / (ML20M["users"] * ML20M["items"])
    lo, hi = sorted((f_int, f_eval))
    log(f"  per-epoch scaling vs ML-20M: interactions {f_int:.2f}x, users x items {f_eval:.2f}x")
    for label, h in (("CUDA-class", ML20M["gpu_hours_per_run_cuda"]),
                     ("Apple-silicon", ML20M["gpu_hours_per_run_mps"])):
        log(f"  20 runs (4 cfg x 5 seeds), {label:<14} {20*h*lo:>6.0f} - {20*h*hi:>5.0f} GPU-hours")

    json.dump({
        "dataset": "steam", "k_core": a.k_core, "min_hours": a.min_hours,
        "positivity": "review exists" + (f" and hours>={a.min_hours}" if a.min_hours else ""),
        "raw": {"interactions": raw_n, "users": raw_u, "items": raw_i},
        "post_kcore": {"interactions": n, "users": nu, "items": ni, "density": dens,
                       "int_per_item": n / ni, "int_per_user": n / nu},
        "split": {"train": tr, "val": va, "test": te},
        "metadata": {"matched": matched, "has_description": False,
                     "coverage_pct": {k: 100 * v / matched for k, v in present.items()},
                     "median_payload_chars": med, "est_input_tokens": est_tok,
                     "median_tags": statistics.median(ntags)},
        "projected": {"generation_usd": ni * per,
                      "gpu_hours_cuda": [20 * ML20M["gpu_hours_per_run_cuda"] * lo,
                                         20 * ML20M["gpu_hours_per_run_cuda"] * hi]},
    }, open(a.out, "w"), indent=2)
    log(f"\n  wrote {a.out}\n")


if __name__ == "__main__":
    main()
