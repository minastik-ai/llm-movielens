"""scripts/regenerate_consolidated.py reads the annotator sheets in both layouts.

The annotators rated blinded sheets (profile_a_<axis>, profile_b_<axis>), and the
release ships them that way, with the decode in annotator_sheet_key.json. The
development copy holds them decoded, one column per model (claude_<axis>,
gpt4o_<axis>). The script read only the decoded layout, so on the released sheets
it stopped with a KeyError, and it looked for them in human_eval/ while
download_artifacts.py writes data/human_eval/. These tests pin the fix on small
synthetic sheets, so they need no download:

  - the two layouts give byte-identical output, and a sheet carrying both is read
    from its decoded columns;
  - the key decides which column is which model, not column order;
  - a blinded sheet without its key, with a key naming another model, or with a
    key describing other sheets stops;
  - sheets that carry neither layout, list the items in different orders, lack
    primary_genre, or have no rows stop;
  - a key that is empty, is not JSON, is not an object, names a thing twice, is nested past what
    json can read, or maps to the wrong things stops and names itself;
  - a rating that is blank, cut short, outside 1 to 5 or written in any form but the digit stops and
    names the file, the row and the column, in the column names the FILE has (profile_a_<axis> on a
    blinded sheet); so does a sheet of zero bytes, one cut in the middle of
    a multi-byte character, one that is not UTF-8, one with a field past the csv size limit, one that
    cannot be opened, one with a repeated or missing column, and one with a row that lost fields;
  - a sheet cut short AFTER its last rating, inside the last column of the last row (the released
    layout ends with primary_genre), either gives the published output or stops: the three sheets
    list the same items, so what identifies an item must agree between them, and a stop that says
    they do not names the two files, and the row where they first differ;
  - a sheet cut at a row boundary, so that it has fewer rows than the key says, is called cut short
    and not blamed on the key alone, and one with MORE rows is not called cut short; a decoded sheet
    that lists fewer items than another is said to be the shorter; and the advice for a file that
    arrived cut is to delete it before fetching again, because the downloader skips files that exist;
  - an output directory that is a file, is under one or cannot be written to stops with a message
    and no traceback, and a stop on the write itself says the two files there may be missing, stale
    or incomplete;
  - ratings so uniform that a kappa is undefined stop before anything is written, and name the
    column the sheet has;
  - a UTF-8 byte-order mark, which an Excel re-save adds, is read;
  - with no arguments, the sheets are found in data/human_eval/, even beside a
    stray human_eval/ that does not hold them;
  - with no sheets, the script exits 2 and names every place it looked.
"""
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _layout import REPO  # noqa: E402

SCRIPT = REPO / "scripts" / "regenerate_consolidated.py"
if not SCRIPT.exists():
    pytest.skip("scripts/regenerate_consolidated.py not present", allow_module_level=True)

AXES = ["thematic_accuracy", "discriminativeness", "rule_compliance",
        "factual_consistency", "coherence_fluency"]
RATERS = ["A1", "A2", "A3"]
N = 8


def rating(i, a, src, ax):
    """Varied but deterministic 1-5 ratings, so every kappa is defined.

    The two models must be rated differently, or a sheet decoded the wrong way
    round would produce the same output and the key test would prove nothing
    (a coefficient of 5 on `src` vanished mod 5 and hid exactly that).
    """
    return 1 + (7 * i + 3 * a + 2 * src + 11 * ax) % 5


def write_sheets(d: Path, blinded: bool, profile_a: str = "claude", order=None,
                 both=False, drop=(), encoding="utf-8"):
    """Three filled sheets, decoded or blinded with profile_a = `profile_a`.

    `both` adds blinded columns holding the OTHER model's ratings to a decoded
    sheet, so reading the wrong set changes the output. `drop` removes columns.
    """
    d.mkdir(parents=True, exist_ok=True)
    srcs = {"claude": 0, "gpt4o": 1}
    for a, rater in enumerate(RATERS):
        rows = []
        for i in (order or range(N)) if a == 0 else range(N):
            base = {"eval_id": str(i + 1), "movieId": str(100 + i), "title": f"Film {i}",
                    "primary_genre": ["drama", "comedy"][i % 2]}
            r = {s: {ax: rating(i, a, k, j) for j, ax in enumerate(AXES)} for s, k in srcs.items()}
            if blinded:
                b = "gpt4o" if profile_a == "claude" else "claude"
                row = {**base, "profile_a_text": "x", "profile_b_text": "y"}
                row.update({f"profile_a_{ax}": r[profile_a][ax] for ax in AXES})
                row.update({f"profile_b_{ax}": r[b][ax] for ax in AXES})
            else:
                row = {**base, "genres": "Drama", "claude_profile_text": "x", "gpt4o_profile_text": "y"}
                row.update({f"{s}_{ax}": r[s][ax] for s in srcs for ax in AXES})
                if both:
                    row.update({f"profile_a_{ax}": r["gpt4o"][ax] for ax in AXES})
                    row.update({f"profile_b_{ax}": r["claude"][ax] for ax in AXES})
            row["notes"] = ""
            rows.append({k: v for k, v in row.items() if k not in drop})
        with open(d / f"annotator_sheet_filled_{rater}.csv", "w", newline="", encoding=encoding) as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def write_key(d: Path, profile_a="claude-haiku-4-5", profile_b="gpt-4o-mini",
              encoding="utf-8", **extra):
    key = {"mapping": {"profile_a": profile_a, "profile_b": profile_b}, "rows": N, "axes": AXES}
    (d / "annotator_sheet_key.json").write_text(json.dumps({**key, **extra}), encoding=encoding)


def run(*args, cwd=None, script=SCRIPT):
    return subprocess.run([sys.executable, str(script), *map(str, args)],
                          capture_output=True, text=True, cwd=cwd)


def outputs(d: Path):
    return [(d / n).read_bytes() for n in ("per_annotator_ratings.csv", "summary_table5.csv")]


def stops(r, text, code=1):
    """A deliberate stop: the expected exit code and message, and no traceback."""
    return r.returncode == code and text in r.stderr and "Traceback" not in r.stderr


def reference(tmp_path):
    """The output of decoded sheets, which every accepted variant must reproduce."""
    write_sheets(tmp_path / "decoded", blinded=False)
    assert run("--human-eval-dir", tmp_path / "decoded", "--outdir", tmp_path / "ref").returncode == 0
    return outputs(tmp_path / "ref")


def copy_script(root: Path) -> Path:
    (root / "scripts").mkdir(parents=True)
    return shutil.copy(SCRIPT, root / "scripts" / SCRIPT.name)


def test_both_layouts_give_identical_output(tmp_path):
    write_sheets(tmp_path / "blind", blinded=True)
    write_key(tmp_path / "blind")
    r = run("--human-eval-dir", tmp_path / "blind", "--outdir", tmp_path / "out")
    assert r.returncode == 0, r.stderr
    assert outputs(tmp_path / "out") == reference(tmp_path)


def test_a_sheet_with_both_layouts_is_read_from_its_decoded_columns(tmp_path):
    write_sheets(tmp_path / "both", blinded=False, both=True)
    write_key(tmp_path / "both")
    r = run("--human-eval-dir", tmp_path / "both", "--outdir", tmp_path / "out")
    assert r.returncode == 0, r.stderr
    assert outputs(tmp_path / "out") == reference(tmp_path)


def test_the_key_decides_which_column_is_which_model(tmp_path):
    write_sheets(tmp_path / "swapped", blinded=True, profile_a="gpt4o")
    write_key(tmp_path / "swapped", profile_a="gpt-4o-mini", profile_b="claude-haiku-4-5")
    assert run("--human-eval-dir", tmp_path / "swapped", "--outdir", tmp_path / "out").returncode == 0
    assert outputs(tmp_path / "out") == reference(tmp_path)


def test_blinded_sheets_without_their_key_stop(tmp_path):
    write_sheets(tmp_path, blinded=True)
    r = run("--human-eval-dir", tmp_path, "--outdir", tmp_path / "out")
    assert stops(r, "is blinded, and the key that decodes it is missing")
    assert "if the download stopped before it, run python3 scripts/download_artifacts.py --human-eval" in r.stderr
    assert not (tmp_path / "out" / "summary_table5.csv").exists()


def test_a_key_naming_another_model_stops(tmp_path):
    write_sheets(tmp_path, blinded=True)
    write_key(tmp_path, profile_b="some-other-model")
    r = run("--human-eval-dir", tmp_path, "--outdir", tmp_path / "out")
    assert stops(r, "must map")


@pytest.mark.parametrize("body", [
    b"", b"{", b"[]", b'"text"', b'{"mapping": ["profile_a", "profile_b"]}',
    b'{"mapping": {"profile_a": 1, "profile_b": "gpt-4o-mini"}}',
    b'{"mapping": {"profile_a": null, "profile_b": null}}'])
def test_a_key_that_is_empty_or_not_a_key_stops(tmp_path, body):
    # an interrupted download leaves an empty or truncated key, and a stranger may hand-edit it
    write_sheets(tmp_path, blinded=True)
    (tmp_path / "annotator_sheet_key.json").write_bytes(body)
    r = run("--human-eval-dir", tmp_path, "--outdir", tmp_path / "out")
    assert stops(r, "annotator_sheet_key.json")
    assert not (tmp_path / "out" / "summary_table5.csv").exists()


@pytest.mark.parametrize("body", [b"", b"{"])
def test_a_key_that_arrived_cut_short_says_to_delete_it_before_fetching_again(tmp_path, body):
    # download_artifacts.py fetches only the files that are missing, so "fetch it again" over a
    # cut file on disk would change nothing
    write_sheets(tmp_path, blinded=True)
    (tmp_path / "annotator_sheet_key.json").write_bytes(body)
    r = run("--human-eval-dir", tmp_path, "--outdir", tmp_path / "out")
    assert stops(r, "cannot be read as JSON")
    assert ("delete it, then run python3 scripts/download_artifacts.py --human-eval "
            "(it fetches only the files that are missing, "
            "into data/human_eval/; add --dry-run to see which)") in r.stderr


def _edit_sheet(path: Path, edit):
    """Rewrite one sheet after `edit(rows)` has changed its rows (a list of dicts)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    edit(rows)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


@pytest.mark.parametrize("value", ["", "x", "6", "0", "4.5", " ", " 4", "4 ", "+4", "04", "\u0664",
                                   "\uff14"])
def test_a_rating_that_is_not_one_to_five_stops(tmp_path, value):
    write_sheets(tmp_path, blinded=False)
    _edit_sheet(tmp_path / "annotator_sheet_filled_A2.csv",
                lambda rows: rows[2].__setitem__("claude_coherence_fluency", value))
    r = run("--human-eval-dir", tmp_path, "--outdir", tmp_path / "out")
    assert stops(r, "annotator_sheet_filled_A2.csv row 4: claude_coherence_fluency is")
    assert "not a rating from 1 to 5" in r.stderr
    assert not (tmp_path / "out" / "summary_table5.csv").exists()


def test_a_sheet_cut_short_mid_row_stops(tmp_path):
    # an interrupted download: the last line stops before its rating cells
    write_sheets(tmp_path, blinded=True)
    write_key(tmp_path)
    path = tmp_path / "annotator_sheet_filled_A1.csv"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[-1] = lines[-1][:40] + "\n"
    path.write_text("".join(lines), encoding="utf-8")
    r = run("--human-eval-dir", tmp_path, "--outdir", tmp_path / "out")
    assert stops(r, "annotator_sheet_filled_A1.csv row") and "incomplete" in r.stderr


@pytest.mark.parametrize("damage", ["cut mid-character", "not utf-8", "a field past the csv limit"])
def test_a_sheet_that_cannot_be_read_as_utf8_csv_stops(tmp_path, damage):
    # an interrupted download ends anywhere, and the released sheets hold non-ASCII text
    write_sheets(tmp_path, blinded=True)
    write_key(tmp_path)
    path = tmp_path / "annotator_sheet_filled_A1.csv"
    _edit_sheet(path, lambda rows: rows[-1].__setitem__(
        "notes", "x" * 200_000 if damage.startswith("a field") else "caf\u00e9"))
    data = path.read_bytes()
    if damage == "cut mid-character":
        data = data[:data.rindex("\u00e9".encode()) + 1]
    elif damage == "not utf-8":
        data = data.replace("\u00e9".encode(), b"\xe9")
    path.write_bytes(data)
    r = run("--human-eval-dir", tmp_path, "--outdir", tmp_path / "out")
    assert stops(r, "annotator_sheet_filled_A1.csv cannot be read") and "incomplete" in r.stderr
    assert not (tmp_path / "out" / "summary_table5.csv").exists()


def _put_last(path: Path, column: str):
    """Rewrite one sheet with `column` as its LAST column, as the released sheets have primary_genre."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields, rows = [c for c in reader.fieldnames if c != column] + [column], list(reader)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_a_sheet_cut_after_its_last_rating_never_gives_a_wrong_output(tmp_path):
    # An interrupted download that ends inside the last column of the last row leaves every rating
    # in place and a shortened value, which is still a value. The output copies primary_genre from
    # A1's sheet alone, so A2's and A3's are what say it was cut.
    write_sheets(tmp_path / "ref", blinded=True)
    write_key(tmp_path / "ref")
    for a in RATERS:
        _put_last(tmp_path / "ref" / f"annotator_sheet_filled_{a}.csv", "primary_genre")
    assert run("--human-eval-dir", tmp_path / "ref", "--outdir", tmp_path / "ref_out").returncode == 0
    want = outputs(tmp_path / "ref_out")
    full = (tmp_path / "ref" / "annotator_sheet_filled_A1.csv").read_bytes()
    stopped = 0
    for k in range(0, len(b"comedy\n") + 8):
        d = tmp_path / f"cut{k}"
        shutil.copytree(tmp_path / "ref", d)
        (d / "annotator_sheet_filled_A1.csv").write_bytes(full[:len(full) - k])
        r = run("--human-eval-dir", d, "--outdir", d / "out")
        if r.returncode == 0:
            assert outputs(d / "out") == want, f"{k} bytes cut off the end: a wrong output was accepted"
        else:
            assert stops(r, "annotator_sheet_filled_A1.csv row") or stops(r, "disagree"), (
                f"{k} bytes cut off the end: {r.stderr!r}")
            stopped += 1
    assert stopped >= 8, "a cut inside the last column must be caught, not only a cut that loses a rating"


def test_sheets_that_disagree_on_what_identifies_an_item_stop(tmp_path):
    write_sheets(tmp_path, blinded=True)
    write_key(tmp_path)
    _edit_sheet(tmp_path / "annotator_sheet_filled_A3.csv",
                lambda rows: rows[3].__setitem__("primary_genre", "drama?"))
    r = run("--human-eval-dir", tmp_path, "--outdir", tmp_path / "out")
    assert stops(r, "annotator_sheet_filled_A3.csv and annotator_sheet_filled_A1.csv disagree at row 5")
    # each value is tied to ITS file: the third rater's sheet holds the damaged one, the first rater's
    # the genre the output would copy
    assert ("primary_genre is 'drama?' in annotator_sheet_filled_A3.csv and 'comedy' in "
            "annotator_sheet_filled_A1.csv;") in r.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("damage", ["a row that lost its tail", "an empty primary_genre",
                                    "an empty eval_id", "no movieId column", "no eval_id column",
                                    "a repeated column"])
def test_a_sheet_that_lost_what_identifies_an_item_stops(tmp_path, damage):
    write_sheets(tmp_path, blinded=True)
    write_key(tmp_path)
    path = tmp_path / "annotator_sheet_filled_A1.csv"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if damage == "a row that lost its tail":
        lines[3] = lines[3][:12] + "\n"
    elif damage == "an empty primary_genre":
        _edit_sheet(path, lambda rows: rows[2].__setitem__("primary_genre", ""))
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    elif damage == "an empty eval_id":
        _edit_sheet(path, lambda rows: rows[2].__setitem__("eval_id", " "))
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    elif damage.startswith("no "):
        column = damage.split()[1]
        lines[0] = lines[0].replace(column, "renamed_" + column, 1)
    else:
        lines[0] = lines[0].replace("notes", "primary_genre", 1)
    path.write_text("".join(lines), encoding="utf-8")
    r = run("--human-eval-dir", tmp_path, "--outdir", tmp_path / "out")
    assert stops(r, "annotator_sheet_filled_A1.csv")
    assert not (tmp_path / "out").exists(), "a stop must leave no half-written output"


def test_a_key_that_names_a_thing_twice_stops(tmp_path):
    # json keeps the LAST of a repeated name, so a key saying mapping twice was read as the second
    write_sheets(tmp_path, blinded=True)
    good = '{"profile_a": "claude-haiku-4-5", "profile_b": "gpt-4o-mini"}'
    swapped = '{"profile_a": "gpt-4o-mini", "profile_b": "claude-haiku-4-5"}'
    (tmp_path / "annotator_sheet_key.json").write_text(
        f'{{"mapping": {good}, "mapping": {swapped}, "rows": {N}, "axes": {json.dumps(AXES)}}}')
    r = run("--human-eval-dir", tmp_path, "--outdir", tmp_path / "out")
    assert stops(r, "annotator_sheet_key.json") and "appears twice" in r.stderr


def test_a_key_nested_past_what_json_reads_stops(tmp_path):
    write_sheets(tmp_path, blinded=True)
    (tmp_path / "annotator_sheet_key.json").write_bytes(b"[" * 200_000)
    r = run("--human-eval-dir", tmp_path, "--outdir", tmp_path / "out")
    assert stops(r, "annotator_sheet_key.json")


@pytest.mark.skipif(not hasattr(__import__("os"), "geteuid") or __import__("os").geteuid() == 0,
                    reason="root can read a file whatever its mode")
def test_a_sheet_that_cannot_be_opened_stops(tmp_path):
    write_sheets(tmp_path, blinded=False)
    path = tmp_path / "annotator_sheet_filled_A2.csv"
    path.chmod(0)
    try:
        r = run("--human-eval-dir", tmp_path, "--outdir", tmp_path / "out")
    finally:
        path.chmod(0o644)
    assert stops(r, "annotator_sheet_filled_A2.csv cannot be read")


def test_ratings_too_uniform_for_a_kappa_stop_before_anything_is_written(tmp_path):
    write_sheets(tmp_path, blinded=False)
    for a in RATERS:
        _edit_sheet(tmp_path / f"annotator_sheet_filled_{a}.csv",
                    lambda rows: [r.__setitem__("claude_coherence_fluency", "3") for r in rows])
    r = run("--human-eval-dir", tmp_path, "--outdir", tmp_path / "out")
    assert stops(r, "the ratings in claude_coherence_fluency are too uniform") and "nothing was written" in r.stderr
    assert not (tmp_path / "out").exists()


def test_two_constant_raters_are_enough_to_stop_the_kappa_and_the_message_says_two(tmp_path):
    # The first two raters gave every item a 3 and the third did not: the weighted kappa of that pair
    # divides by zero. The message must name the two raters, not say that EVERY rater gave every item
    # the same rating
    write_sheets(tmp_path, blinded=False)
    for a in ("A1", "A2"):
        _edit_sheet(tmp_path / f"annotator_sheet_filled_{a}.csv",
                    lambda rows: [r.__setitem__("claude_coherence_fluency", "3") for r in rows])
    r = run("--human-eval-dir", tmp_path, "--outdir", tmp_path / "out")
    assert stops(r, "the ratings in claude_coherence_fluency are too uniform for a kappa (at least two of the "
                    "raters gave every item the same rating, or there are too few items)")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("profile_a, model_a, model_b, column", [
    ("claude", "claude-haiku-4-5", "gpt-4o-mini", "profile_a_coherence_fluency"),
    ("gpt4o", "gpt-4o-mini", "claude-haiku-4-5", "profile_b_coherence_fluency")])
def test_uniform_ratings_in_a_blinded_sheet_name_the_column_the_file_has(
        tmp_path, profile_a, model_a, model_b, column):
    # the decoded name (claude_coherence_fluency) is not in the released sheets; which of the two
    # blinded columns holds claude's ratings is the key's to say, so the swapped key moves it
    write_sheets(tmp_path, blinded=True, profile_a=profile_a)
    write_key(tmp_path, profile_a=model_a, profile_b=model_b)
    for a in RATERS:
        _edit_sheet(tmp_path / f"annotator_sheet_filled_{a}.csv",
                    lambda rows: [r.__setitem__(column, "3") for r in rows])
    r = run("--human-eval-dir", tmp_path, "--outdir", tmp_path / "out")
    assert stops(r, f"the ratings in {column} are too uniform")
    assert "claude_" not in r.stderr and "gpt4o_" not in r.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("extra, said", [
    ({"rows": N + 1}, f"or annotator_sheet_filled_A1.csv is cut short: the key says {N + 1} rows and the sheet has {N};"),
    ({"axes": list(reversed(AXES))}, f"its axes are {list(reversed(AXES))!r}")])
def test_a_key_describing_other_sheets_stops(tmp_path, extra, said):
    write_sheets(tmp_path, blinded=True)
    write_key(tmp_path, **extra)
    r = run("--human-eval-dir", tmp_path, "--outdir", tmp_path / "out")
    assert stops(r, "annotator_sheet_key.json describes other sheets") and said in r.stderr


def test_a_sheet_with_more_rows_than_the_key_says_is_not_called_cut_short(tmp_path):
    # a duplicated row (a bad copy-paste, a merge) makes the sheet LONGER than the key says; the
    # message must not call that "cut short", which would send a stranger looking for a short download
    write_sheets(tmp_path, blinded=True)
    write_key(tmp_path, rows=N - 1)
    r = run("--human-eval-dir", tmp_path, "--outdir", tmp_path / "out")
    assert stops(r, "or annotator_sheet_filled_A1.csv has rows added or repeated: "
                    f"the key says {N - 1} rows and the sheet has {N};")
    assert "cut short" not in r.stderr
    assert ("delete both files, then run python3 scripts/download_artifacts.py --human-eval "
            "(it fetches only the files that are missing, "
            "into data/human_eval/; add --dry-run to see which)") in r.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("rows", [str(N), float(N - 1), None, True])
def test_a_key_whose_row_count_is_not_a_whole_number_says_so(tmp_path, rows):
    # "8" and 8 print the same, so a message reading "8 rows; the sheet has 8 rows" says nothing; a count
    # that is not a whole number has no direction either, so it is never "cut short" or "added to"
    write_sheets(tmp_path, blinded=True)
    write_key(tmp_path, rows=rows)
    r = run("--human-eval-dir", tmp_path, "--outdir", tmp_path / "out")
    assert stops(r, f"annotator_sheet_key.json must give the number of rows as a whole number, not {rows!r} "
                    f"(the sheet annotator_sheet_filled_A1.csv has {N})")
    assert "cut short" not in r.stderr and "added or repeated" not in r.stderr
    assert not (tmp_path / "out").exists()


def test_a_key_without_a_row_count_is_accepted(tmp_path):
    # `rows` is optional in the key; its absence must not read as a mismatch
    write_sheets(tmp_path, blinded=True)
    (tmp_path / "annotator_sheet_key.json").write_text(json.dumps(
        {"mapping": {"profile_a": "claude-haiku-4-5", "profile_b": "gpt-4o-mini"}, "axes": AXES}))
    r = run("--human-eval-dir", tmp_path, "--outdir", tmp_path / "out")
    assert r.returncode == 0, r.stderr
    assert outputs(tmp_path / "out") == reference(tmp_path)


def test_a_sheet_cut_at_a_row_boundary_is_called_cut_short(tmp_path):
    # a download that ends between two rows leaves whole rows and fewer of them than the key says;
    # the message must not blame the key alone, or a stranger would fetch the key again
    write_sheets(tmp_path, blinded=True)
    write_key(tmp_path)
    path = tmp_path / "annotator_sheet_filled_A1.csv"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(lines[:4]), encoding="utf-8")           # the header and three whole rows
    r = run("--human-eval-dir", tmp_path, "--outdir", tmp_path / "out")
    assert stops(r, "annotator_sheet_filled_A1.csv is cut short")
    assert f"the key says {N} rows and the sheet has 3" in r.stderr
    # download_artifacts.py skips a file that is already there, so the cut one has to go first
    assert ("delete both files, then run python3 scripts/download_artifacts.py --human-eval "
            "(it fetches only the files that are missing, "
            "into data/human_eval/; add --dry-run to see which)") in r.stderr
    assert not (tmp_path / "out").exists()


def test_a_bad_rating_names_the_column_the_swapped_key_puts_the_model_in(tmp_path):
    # with profile_a = gpt-4o-mini, claude's ratings sit in profile_b: a message that named the
    # column by position instead of through the key would send a stranger to the other one
    write_sheets(tmp_path, blinded=True, profile_a="gpt4o")
    write_key(tmp_path, profile_a="gpt-4o-mini", profile_b="claude-haiku-4-5")
    _edit_sheet(tmp_path / "annotator_sheet_filled_A2.csv",
                lambda rows: rows[2].__setitem__("profile_b_discriminativeness", "9"))
    r = run("--human-eval-dir", tmp_path, "--outdir", tmp_path / "out")
    assert stops(r, "annotator_sheet_filled_A2.csv row 4: profile_b_discriminativeness is '9'")
    assert "profile_a_discriminativeness" not in r.stderr and not (tmp_path / "out").exists()


def test_a_bad_rating_in_a_blinded_sheet_names_the_column_the_file_has(tmp_path):
    # the message must name the column the file has, not the decoded one (gpt4o_rule_compliance), which
    # the released sheets do not contain
    write_sheets(tmp_path, blinded=True)
    write_key(tmp_path)
    _edit_sheet(tmp_path / "annotator_sheet_filled_A1.csv",
                lambda rows: rows[4].__setitem__("profile_b_rule_compliance", ""))
    r = run("--human-eval-dir", tmp_path, "--outdir", tmp_path / "out")
    assert stops(r, "annotator_sheet_filled_A1.csv row 6: profile_b_rule_compliance is ''")
    assert "gpt4o_" not in r.stderr and "claude_" not in r.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("where", ["a file", "under a file"])
def test_an_output_directory_that_cannot_be_made_stops(tmp_path, where):
    write_sheets(tmp_path / "in", blinded=False)
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    out = blocker if where == "a file" else blocker / "out"
    r = run("--human-eval-dir", tmp_path / "in", "--outdir", out)
    # the directory the stranger gave, the error that says why, and what the stop may have left
    assert stops(r, f"cannot write the output to {out} (")
    assert ("FileExistsError" if where == "a file" else "NotADirectoryError") in r.stderr
    assert "may be missing, stale or incomplete" in r.stderr
    assert blocker.read_text() == "not a directory"


@pytest.mark.skipif(not hasattr(__import__("os"), "geteuid") or __import__("os").geteuid() == 0,
                    reason="root can write wherever it likes")
def test_an_output_directory_that_cannot_be_written_to_stops(tmp_path):
    write_sheets(tmp_path / "in", blinded=False)
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        r = run("--human-eval-dir", tmp_path / "in", "--outdir", locked)
    finally:
        locked.chmod(0o700)
    assert stops(r, f"cannot write the output to {locked} (PermissionError")
    assert list(locked.iterdir()) == []


@pytest.mark.skipif(not hasattr(__import__("os"), "geteuid") or __import__("os").geteuid() == 0,
                    reason="root can write wherever it likes")
def test_a_stop_on_the_write_itself_says_the_output_may_be_incomplete(tmp_path):
    # The one stop that can leave something behind: the first file is written, the second cannot be
    # (here a read-only summary from an earlier run). Nothing is promised that is not true.
    write_sheets(tmp_path / "in", blinded=False)
    out = tmp_path / "out"
    out.mkdir()
    old = out / "summary_table5.csv"
    old.write_text("stale\n")
    old.chmod(0o400)
    try:
        r = run("--human-eval-dir", tmp_path / "in", "--outdir", out)
    finally:
        old.chmod(0o600)
    assert stops(r, f"cannot write the output to {out} (PermissionError")
    assert "the two files there may be missing, stale or incomplete" in r.stderr
    assert old.read_text() == "stale\n"
    assert (out / "per_annotator_ratings.csv").is_file()     # why the message cannot say "nothing was written"


@pytest.mark.skipif(not hasattr(__import__("os"), "geteuid") or __import__("os").geteuid() == 0,
                    reason="root can write wherever it likes")
def test_a_write_failure_without_outdir_names_the_directory_the_script_chose(tmp_path):
    # without --outdir the output goes to consolidated/ beside the sheets, and that is what the message
    # must name (it once could have printed the unset option)
    write_sheets(tmp_path / "in", blinded=False)
    (tmp_path / "in").chmod(0o500)
    try:
        r = run("--human-eval-dir", tmp_path / "in")
    finally:
        (tmp_path / "in").chmod(0o700)
    assert stops(r, f"cannot write the output to {tmp_path / 'in' / 'consolidated'} (PermissionError")


def test_a_sheet_with_neither_layout_stops(tmp_path):
    write_sheets(tmp_path, blinded=False, drop=tuple(f"claude_{ax}" for ax in AXES))
    r = run("--human-eval-dir", tmp_path, "--outdir", tmp_path / "out")
    assert stops(r, "has neither the decoded rating columns")


def test_sheets_in_different_item_orders_stop(tmp_path):
    write_sheets(tmp_path, blinded=False, order=list(reversed(range(N))))
    r = run("--human-eval-dir", tmp_path, "--outdir", tmp_path / "out")
    assert stops(r, "annotator_sheet_filled_A2.csv does not list the same items in the same order "
                    "as annotator_sheet_filled_A1.csv: they first differ at row 2")


def test_the_row_where_the_item_order_first_differs_is_named(tmp_path):
    write_sheets(tmp_path, blinded=False)
    _edit_sheet(tmp_path / "annotator_sheet_filled_A3.csv",
                lambda rows: rows.__setitem__(slice(4, 6), [rows[5], rows[4]]))
    r = run("--human-eval-dir", tmp_path, "--outdir", tmp_path / "out")
    assert stops(r, "annotator_sheet_filled_A3.csv does not list the same items") and \
        "they first differ at row 6" in r.stderr
    assert "items and" not in r.stderr          # the sheets are the same length: nobody is "shorter"


@pytest.mark.parametrize("cut, compared, said", [
    ("A3", "A3", "(annotator_sheet_filled_A3.csv lists 7 items and annotator_sheet_filled_A1.csv lists 8)"),
    ("A1", "A2", "(annotator_sheet_filled_A2.csv lists 8 items and annotator_sheet_filled_A1.csv lists 7)")])
def test_a_sheet_that_lists_fewer_items_is_said_to_be_shorter(tmp_path, cut, compared, said):
    # Decoded sheets have no key to count rows against, so a sheet cut at a row boundary is found
    # by comparing item lists: the rows it has agree, and the first row it lacks (header = row 1,
    # seven items = rows 2 to 8) is where they differ. The message says who is shorter.
    write_sheets(tmp_path, blinded=False)
    _edit_sheet(tmp_path / f"annotator_sheet_filled_{cut}.csv",
                lambda rows: rows.__delitem__(slice(N - 1, None)))
    r = run("--human-eval-dir", tmp_path, "--outdir", tmp_path / "out")
    assert stops(r, f"annotator_sheet_filled_{compared}.csv does not list the same items in the "
                    f"same order as annotator_sheet_filled_A1.csv: they first differ at row {N + 1} ")
    assert said in r.stderr and not (tmp_path / "out").exists()


def test_sheets_without_primary_genre_stop(tmp_path):
    write_sheets(tmp_path, blinded=False, drop=("primary_genre",))
    r = run("--human-eval-dir", tmp_path, "--outdir", tmp_path / "out")
    assert stops(r, "has no primary_genre column")


def test_a_sheet_with_a_header_and_no_rows_says_so(tmp_path):
    write_sheets(tmp_path, blinded=False)
    with open(tmp_path / "annotator_sheet_filled_A1.csv") as f:
        header = f.readline()
    (tmp_path / "annotator_sheet_filled_A1.csv").write_text(header)
    r = run("--human-eval-dir", tmp_path, "--outdir", tmp_path / "out")
    assert stops(r, "has no rows")
    # and a sheet of ZERO bytes, which a failed download leaves behind: the header is read
    # lazily there, and reading it after the file was closed raised a ValueError traceback
    (tmp_path / "annotator_sheet_filled_A1.csv").write_bytes(b"")
    r = run("--human-eval-dir", tmp_path, "--outdir", tmp_path / "out")
    assert stops(r, "has no rows")


def test_a_byte_order_mark_is_read(tmp_path):
    write_sheets(tmp_path / "bom", blinded=True, encoding="utf-8-sig")
    write_key(tmp_path / "bom", encoding="utf-8-sig")
    r = run("--human-eval-dir", tmp_path / "bom", "--outdir", tmp_path / "out")
    assert r.returncode == 0, r.stderr
    assert outputs(tmp_path / "out") == reference(tmp_path)


def test_finds_the_sheets_where_download_artifacts_puts_them(tmp_path):
    script = copy_script(tmp_path)
    (tmp_path / "human_eval").mkdir()      # a stray directory that does not hold them
    write_sheets(tmp_path / "data" / "human_eval", blinded=True)
    write_key(tmp_path / "data" / "human_eval")
    r = run(script=script, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "data" / "human_eval" / "consolidated" / "summary_table5.csv").exists()


def test_no_sheets_exits_2_and_names_every_place_it_looked(tmp_path):
    script = copy_script(tmp_path)
    r = run(script=script, cwd=tmp_path)
    assert stops(r, "download_artifacts.py --human-eval", code=2)
    for place in (tmp_path / "human_eval", tmp_path / "data" / "human_eval"):
        assert str(place.resolve()) in r.stderr


def test_an_absent_suffixed_set_is_not_said_to_ship(tmp_path):
    write_sheets(tmp_path, blinded=False)
    r = run("--sheets", "_v0", "--human-eval-dir", tmp_path, "--outdir", tmp_path / "out")
    assert stops(r, "annotator_sheet_filled_A1_v0.csv", code=2)
    assert "download_artifacts" not in r.stderr
