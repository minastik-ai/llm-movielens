#!/usr/bin/env python3
"""Regenerate consolidated/, beside the filled annotator sheets, from those sheets.

Emits two files, with their existing schemas:

  per_annotator_ratings.csv   long form, one row per (item, annotator, source, axis)
  summary_table5.csv          per (source, axis): mean, std, Fleiss kappa,
                              linearly-weighted kappa, >=4 rate

The annotators rated blinded sheets: a film's two profiles sit in columns named
profile_a and profile_b, and annotator_sheet_key.json, beside the sheets, says
which model each column holds. Sheets in that layout are decoded through the
key, which is required; the mapping is never guessed from column order. Sheets
already decoded, with one column per model and axis (claude_<axis>,
gpt4o_<axis>), are read as they are, and a sheet carrying both layouts is read
from its decoded columns.

The sheets are read from --human-eval-dir, or else from the first of human_eval/
and data/human_eval/ that holds them; `scripts/download_artifacts.py --human-eval`
writes data/human_eval/. The output goes to consolidated/ beside the sheets
unless --outdir says otherwise. `--sheets SUFFIX` reads
annotator_sheet_filled_A1SUFFIX.csv and so on instead, to compare two sets of
sheets. The mood sheets are not read here.

The sheets are checked before anything is written. Each must read as UTF-8 CSV, name every
column once, have every column in every row, and carry ratings that are exactly the digits 1
to 5; the three must list the same items and agree on movieId and primary_genre row by row (the
last column of the last row is the one a download that ends early can shorten without a trace,
and the output copies it from A1's sheet alone). A sheet that fails any of this stops the script
with a message that names the file, and the row where a row is meant, in the column names that
file has; nothing is written. Only a failure of the write itself (a full disk, a read-only file
in --outdir) can leave the two output files missing, stale or incomplete, and its message says so.
"""
import argparse, csv, itertools, json, statistics as st, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Beside a clone, or where download_artifacts.py puts them.
CANDIDATES = [ROOT / "human_eval", ROOT / "data" / "human_eval"]
KEY = "annotator_sheet_key.json"
AXES = ["thematic_accuracy", "discriminativeness", "rule_compliance",
        "factual_consistency", "coherence_fluency"]
SOURCES = ["claude", "gpt4o"]
MODELS = {"claude-haiku-4-5": "claude", "gpt-4o-mini": "gpt4o"}  # the key's names
BLIND = ["profile_a", "profile_b"]
RATERS = ["A1", "A2", "A3"]
CATS = [1, 2, 3, 4, 5]
ID_COLS = ["eval_id", "movieId", "primary_genre"]    # what identifies an item in the output
# What to do about a file that arrived cut short: the downloader fetches only the files that are
# missing, so running it again over a cut file changes nothing until the file is deleted.
FETCH = ("python3 scripts/download_artifacts.py --human-eval (it fetches only the files that are missing, "
         "into data/human_eval/; add --dry-run to see which)")


def fail(msg):
    sys.exit(f"regenerate_consolidated: {msg}")


def _unique(pairs):
    """json.loads keeps the LAST of a repeated name: a key that says mapping twice would be read as
    the second one without a word."""
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError(f"the name {k!r} appears twice")
        d[k] = v
    return d


def decode(rows, cols, he, name):
    """(rows, origin): the rows with claude_<axis> and gpt4o_<axis> columns, whichever layout they
    came in, and for each decoded column the column of the FILE it came from (empty for a sheet
    that was already decoded), so that a message names a column the file has."""
    if {f"{s}_{ax}" for s in SOURCES for ax in AXES} <= cols:
        return rows, {}
    if not {f"{b}_{ax}" for b in BLIND for ax in AXES} <= cols:
        fail(f"{name} has neither the decoded rating columns (claude_<axis>, "
             f"gpt4o_<axis>) nor the blinded ones (profile_a_<axis>, profile_b_<axis>)")
    path = he / KEY
    if not path.is_file():
        fail(f"{name} is blinded, and the key that decodes it is missing: {path}; if the download "
             f"stopped before it, run {FETCH}")
    try:
        key = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique)
    except (ValueError, RecursionError, OSError) as e:   # an interrupted download leaves an empty
        fail(f"{path} cannot be read as JSON ({type(e).__name__}: {e}); "   # or cut file
             f"delete it, then run {FETCH}")
    if not isinstance(key, dict):
        fail(f"{path} must hold a JSON object, not a {type(key).__name__}")
    mapping = key.get("mapping")
    mapping = mapping if isinstance(mapping, dict) else {}
    if (sorted(mapping) != BLIND or not all(isinstance(v, str) for v in mapping.values())
            or sorted(mapping.values()) != sorted(MODELS)):
        fail(f"{path} must map {BLIND} to {sorted(MODELS)}; it maps {mapping}")
    if key.get("axes", AXES) != AXES:
        fail(f"{path} describes other sheets: its axes are {key.get('axes')!r}")
    want = key.get("rows", len(rows))
    if want != len(rows):
        if type(want) is not int:            # "500", 499.0, null, true: the key is malformed, not the sheet
            fail(f"{path} must give the number of rows as a whole number, not {want!r} (the sheet "
                 f"{name} has {len(rows)}); delete the key, then run {FETCH}")
        # The direction says what happened to the sheet: a download that ends early leaves fewer rows
        # than the key says, and a sheet with MORE has had rows added or repeated.
        what = f"{name} has rows added or repeated" if len(rows) > want else f"{name} is cut short"
        fail(f"{path} describes other sheets, or {what}: the key says {want!r} rows and the sheet "
             f"has {len(rows)}; delete both files, then run {FETCH}")
    src = {b: MODELS[m] for b, m in mapping.items()}
    origin = {f"{src[b]}_{ax}": f"{b}_{ax}" for b in BLIND for ax in AXES}
    return [{**r, **{d: r[o] for d, o in origin.items()}} for r in rows], origin


def check_ratings(rows, name, origin):
    """Every rating is exactly one of the digits 1 to 5. A sheet cut short by an interrupted
    download, or saved with a blank cell, would otherwise stop the arithmetic with a traceback; and
    a cell in any other form (" 4", "+4", "04", a non-ASCII digit that int() accepts) would be
    copied into the output as it stands. The message names the column the FILE has: on a blinded
    sheet that is profile_a_<axis>, not the claude_<axis> it decodes to."""
    valid = {str(c) for c in CATS}
    for i, r in enumerate(rows, 2):                     # row 1 is the header
        for s in SOURCES:
            for ax in AXES:
                col = f"{s}_{ax}"
                v = r.get(col)
                if v not in valid:
                    fail(f"{name} row {i}: {origin.get(col, col)} is {v!r}, not a rating from 1 "
                         f"to 5; the file may be incomplete")


def check_rows(rows, name):
    """Every row has every column the header names, and the three that identify an item are filled
    in. A download that ends early ends inside some row: csv reads the columns it lost as None,
    and this refuses the row and names the first of them, a rating or a column after the ratings
    alike (it runs before check_ratings, which is left with ratings that are present and wrong)."""
    for i, r in enumerate(rows, 2):                     # row 1 is the header
        if None in r:
            fail(f"{name} row {i}: more fields than the header names; the file may be damaged")
        short = [c for c, v in r.items() if v is None]
        if short:
            fail(f"{name} row {i}: no {short[0]!r} field; the file may be incomplete")
        for c in ID_COLS:
            if not r[c].strip():
                fail(f"{name} row {i}: {c} is empty; the file may be incomplete")


def load(he, suffix=""):
    """(sheets, origins): each rater's rows with the decoded columns, and for each rater the map from
    a decoded column to the column of the file it came from (see decode)."""
    sheets, files, origins = {}, {}, {}
    for a in RATERS:
        path = he / f"annotator_sheet_filled_{a}{suffix}.csv"
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:  # -sig: an Excel re-save adds a BOM
                reader = csv.DictReader(f)
                rows = list(reader)
                # Read inside the block: on an EMPTY file the header is fetched lazily, and after
                # the file is closed that fetch raised ValueError instead of "has no rows".
                cols = set(reader.fieldnames or ())
        except (UnicodeDecodeError, csv.Error, OSError) as e:
            # a download cut in the middle of a multi-byte character (the sheets hold em-dashes and
            # accents), a byte that is not UTF-8, a field past the csv module's size limit, a file
            # that cannot be opened
            fail(f"{path.name} cannot be read as a UTF-8 CSV file ({type(e).__name__}: {e}); "
                 f"the file may be incomplete")
        if not rows:
            fail(f"{path.name} has no rows")
        for c in ID_COLS:
            if c not in cols:
                fail(f"{path.name} has no {c} column")
        names = reader.fieldnames
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            fail(f"{path.name} names the column {repeated[0]!r} more than once; csv would keep "
                 f"the last one")
        check_rows(rows, path.name)
        sheets[a], origins[a] = decode(rows, cols, he, path.name)
        check_ratings(sheets[a], path.name, origins[a])
        files[a] = path.name
    ids = [r["eval_id"] for r in sheets["A1"]]
    for a in RATERS[1:]:
        other = [r["eval_id"] for r in sheets[a]]
        if other != ids:
            first = next((i for i, (x, y) in enumerate(zip(ids, other), 2) if x != y),
                         min(len(ids), len(other)) + 2)      # one list is a prefix of the other
            sizes = (f" ({files[a]} lists {len(other)} items and {files['A1']} lists {len(ids)})"
                     if len(other) != len(ids) else "")
            fail(f"{files[a]} does not list the same items in the same order as {files['A1']}: "
                 f"they first differ at row {first}{sizes}")
        # The last column of the last row is the one a download that ends early can shorten
        # without leaving a trace (a shortened value is still a value), and it is the column the
        # output copies from A1's sheet alone: the other two sheets say what it should be.
        for i, (x, y) in enumerate(zip(sheets["A1"], sheets[a]), 2):
            for c in ID_COLS[1:]:
                if x[c] != y[c]:
                    fail(f"{files[a]} and {files['A1']} disagree at row {i}: {c} is {y[c]!r} in "
                         f"{files[a]} and {x[c]!r} in {files['A1']}; one of the files may be cut "
                         f"short or damaged")
    return sheets, origins


def fleiss(sheets, src, axis):
    n = len(RATERS)
    N = len(sheets["A1"])
    M = [[Counter(int(sheets[a][i][f"{src}_{axis}"]) for a in RATERS).get(j, 0)
          for j in CATS] for i in range(N)]
    Pbar = sum((sum(x * x for x in r) - n) / (n * (n - 1)) for r in M) / N
    pj = [sum(r[j] for r in M) / (N * n) for j in range(len(CATS))]
    Pe = sum(p * p for p in pj)
    return (Pbar - Pe) / (1 - Pe)


def weighted_kappa(sheets, src, axis):
    """Linearly-weighted Cohen's kappa, averaged over the three rater pairs."""
    N = len(sheets["A1"])
    w = lambda p, q: 1 - abs(p - q) / (len(CATS) - 1)
    out = []
    for a, b in itertools.combinations(RATERS, 2):
        x = [int(sheets[a][i][f"{src}_{axis}"]) for i in range(N)]
        y = [int(sheets[b][i][f"{src}_{axis}"]) for i in range(N)]
        Po = sum(w(p, q) for p, q in zip(x, y)) / N
        cx, cy = Counter(x), Counter(y)
        Pe = sum(w(p, q) * cx[p] / N * cy[q] / N for p in CATS for q in CATS)
        out.append((Po - Pe) / (1 - Pe))
    return sum(out) / len(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sheets", default="", metavar="SUFFIX",
                    help="read annotator_sheet_filled_A1SUFFIX.csv and so on "
                         "(default: no suffix)")
    ap.add_argument("--human-eval-dir", type=Path, default=None,
                    help="the directory holding the filled sheets (default: the first "
                         "of human_eval/, data/human_eval/ that holds them)")
    ap.add_argument("--outdir", default=None,
                    help="where to write (default: consolidated/ beside the sheets)")
    args = ap.parse_args()
    # The first place that HOLDS the sheets, not the first that exists:
    # scripts/prepare_human_eval.py creates human_eval/.
    names = [f"annotator_sheet_filled_{a}{args.sheets}.csv" for a in RATERS]
    places = [args.human_eval_dir] if args.human_eval_dir else CANDIDATES
    he = next((d for d in places if all((d / n).is_file() for n in names)), None)
    if he is None:
        print(f"CANNOT RUN: {', '.join(names)} are not all in any of:", file=sys.stderr)
        for d in places:
            print(f"    {d}", file=sys.stderr)
        if not args.sheets:
            print("They ship on the data repository under human_eval/. Fetch them with\n"
                  "    python3 scripts/download_artifacts.py --human-eval\n"
                  "which writes data/human_eval/, where this looks by default, or pass\n"
                  "--human-eval-dir.", file=sys.stderr)
        sys.exit(2)
    S, origins = load(he, args.sheets)
    N = len(S["A1"])
    out = Path(args.outdir) if args.outdir else he / "consolidated"

    # Everything is worked out BEFORE anything is written, so a stop on the sheets leaves nothing. A
    # stop on the write itself (below) is the one that can leave the output missing, stale or cut off.
    long_rows = []
    for a in RATERS:
        for i in range(N):
            base = S["A1"][i]
            for ax in AXES:
                for src in SOURCES:
                    long_rows.append([base["eval_id"], base["movieId"], base["primary_genre"],
                                      a, src, ax, S[a][i][f"{src}_{ax}"]])
    summary = []
    for src in SOURCES:
        for ax in AXES:
            v = [int(S[a][i][f"{src}_{ax}"]) for i in range(N) for a in RATERS]
            try:
                summary.append([src, ax, round(st.mean(v), 3), round(st.stdev(v), 3),
                                round(fleiss(S, src, ax), 3),
                                round(weighted_kappa(S, src, ax), 3),
                                round(sum(1 for x in v if x >= 4) / len(v), 3)])
            except (ZeroDivisionError, st.StatisticsError):
                col = f"{src}_{ax}"        # named as the sheets have it: profile_a_<axis> when blinded
                fail(f"the ratings in {origins['A1'].get(col, col)} are too uniform for a kappa "
                     f"(at least two of the raters gave every item the same rating, or there are too "
                     f"few items); nothing was written")

    try:
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "per_annotator_ratings.csv", "w", newline="") as f:
            wr = csv.writer(f, lineterminator="\n")
            wr.writerow(["eval_id", "movieId", "primary_genre", "annotator_id",
                         "source", "axis", "rating"])
            wr.writerows(long_rows)
        with open(out / "summary_table5.csv", "w", newline="") as f:
            wr = csv.writer(f, lineterminator="\n")
            wr.writerow(["source", "axis", "mean", "std", "fleiss_kappa",
                         "weighted_kappa", "ge4_rate"])
            wr.writerows(summary)
    except OSError as e:        # --outdir is a file, is under one, or cannot be written to; or the disk is full
        fail(f"cannot write the output to {out} ({type(e).__name__}: {e}); the two files there may "
             f"be missing, stale or incomplete")
    print(f"wrote {out}/per_annotator_ratings.csv  ({N * len(RATERS) * len(SOURCES) * len(AXES)} rows)")
    print(f"wrote {out}/summary_table5.csv")


if __name__ == "__main__":
    main()
