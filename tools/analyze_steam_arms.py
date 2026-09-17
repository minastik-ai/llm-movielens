#!/usr/bin/env python3
"""Measure the world-knowledge share in Steam profiles from the paired arms.

THE QUESTION. Steam ships no free-text description: the median prompt payload is
~61 tokens of title, tags, genres, specs and developer. An 80-120 word profile is
therefore roughly twice the length of its own input, so most of the content must
come from somewhere. Either the model is synthesising from the supplied structured
metadata -- which is what the paper claims to measure -- or it is recalling the
game from pretraining, which is a different capability and not a property of the
resource.

THE DESIGN. Generate the same catalogue twice. The FULL arm sees everything. The
MASKED arm has title, developer and publisher withheld, so the identity key that
makes recall possible is gone; only tags, genres and specs remain. Anything the
full arm says that the masked arm cannot is the world-knowledge share.

FOUR MEASUREMENTS, cheapest first:
  1. Lexical grounding per arm -- what share of profile content words appear in
     the supplied metadata. A crude proxy (a model that abstracts "arena shooter"
     from "FPS, Multiplayer" scores zero), which is exactly why it cannot settle
     the question alone. Reported for calibration against ML-20M, not as a verdict.
  2. Identity leakage -- does a MASKED profile still name the game it was not
     told about? If the model recovers identity from tags alone, masking failed as
     a control and the design must be reconsidered before anything is published.
  3. Paired embedding distance -- full vs masked for the SAME game. Near-identical
     means the withheld identity contributed nothing and Steam is safe to use.
  4. Discriminability -- mean pairwise cosine within each arm. If masked profiles
     collapse toward each other, the metadata alone cannot separate the catalogue,
     which is a finding about Steam rather than about the method.

Usage:  python analyze_steam_arms.py [--embed]     (--embed adds 3 and 4)
"""
import argparse, json, re, statistics as st, sys
from pathlib import Path

_HERE = Path(__file__).resolve()
def _find_up(start, *patterns):
    """First ancestor of `start` containing any of `patterns` (globs allowed).
    Names no sibling project: the development checkout is found by SHAPE, so the
    same code runs unchanged from a published clone."""
    for base in [start, *start.parents]:
        for pat in patterns:
            hits = sorted(base.glob(pat))
            if hits:
                return hits[0]
    return None


GEN = (_find_up(_HERE.parent, "output_steam", "*/code/profile_generator/output_steam")
       or _HERE.parents[1] / "output_steam")

STOP = set("""a an the and or but of to in on for with without from by as at is are was were be been
being it its this that these those they them their there here what which who whom how when where why
all any both each few more most other some such no nor not only own same so than too very can will
just don should now into over under again further then once through during before after above below
up down out off you your yours we our us i me my he she his her one two into more into""".split())


def words(s: str) -> set:
    return {w for w in re.findall(r"[a-z]+", s.lower()) if len(w) > 3 and w not in STOP}


def metadata_text(m: dict) -> str:
    parts = [str(v) for v in m.values() if isinstance(v, (str, int, float))]
    for k in ("tags", "genres", "specs"):
        parts += [str(t) for t in (m.get(k) or [])]
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true", help="also run 3 and 4 (loads a model)")
    a = ap.parse_args()

    meta = json.loads((GEN / "game_metadata.json").read_text())
    full = json.loads((GEN / "game_profiles.json").read_text())
    mpath = GEN / "game_profiles_masked.json"
    if not mpath.exists():
        sys.exit(f"masked arm not found at {mpath} -- run with --mask-identity first")
    masked = json.loads(mpath.read_text())

    paired = sorted(set(full) & set(masked), key=lambda k: int(k))
    print(f"full {len(full):,} | masked {len(masked):,} | PAIRED {len(paired):,}")
    if not paired:
        sys.exit("no games appear in both arms -- the pairing is broken, stop here")

    # ---- 1. lexical grounding -------------------------------------------------
    def grounding(store):
        out = []
        for k in paired:
            m = meta.get(str(store[k].get("app_id"))) or meta.get(k) or {}
            pw = words(store[k]["profile"])
            if pw:
                out.append(len(pw & words(metadata_text(m))) / len(pw))
        return out
    gf, gm = grounding(full), grounding(masked)
    print("\n1. lexical grounding in the supplied metadata")
    print(f"   full   median {st.median(gf)*100:5.1f}%   mean {st.mean(gf)*100:5.1f}%")
    print(f"   masked median {st.median(gm)*100:5.1f}%   mean {st.mean(gm)*100:5.1f}%")
    print(f"   ML-20M median  11.4%   (measured on 200 profiles, same method)")

    # ---- 2. identity leakage --------------------------------------------------
    leaked, checked = [], 0
    for k in paired:
        title = (full[k].get("title") or "").strip()
        toks = [t.lower() for t in re.findall(r"[A-Za-z]{4,}", title)
                if t.lower() not in STOP]
        if not toks:
            continue
        checked += 1
        body = masked[k]["profile"].lower()
        if any(t in body for t in toks):
            leaked.append((title, [t for t in toks if t in body]))
    print("\n2. identity leakage in the MASKED arm (the control's own validity)")
    print(f"   {len(leaked)}/{checked} masked profiles echo a distinctive title token "
          f"({100*len(leaked)/max(checked,1):.1f}%)")
    for t, hits in leaked[:5]:
        print(f"     {t}  ->  {hits}")
    if len(leaked) / max(checked, 1) > 0.25:
        print("   WARNING: the model is recovering identity from tags alone. The masked")
        print("   arm is then not a clean control and the gap UNDERSTATES recall.")

    if not a.embed:
        print("\n(3 and 4 skipped; pass --embed)")
        return

    # ---- 3 & 4. embedding geometry -------------------------------------------
    from sentence_transformers import SentenceTransformer
    import numpy as np
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    ef = model.encode([full[k]["profile"] for k in paired], normalize_embeddings=True,
                      batch_size=32, show_progress_bar=False)
    em = model.encode([masked[k]["profile"] for k in paired], normalize_embeddings=True,
                      batch_size=32, show_progress_bar=False)
    pair_cos = (ef * em).sum(1)
    print("\n3. paired cosine, full vs masked, same game")
    print(f"   median {np.median(pair_cos):.3f}   mean {pair_cos.mean():.3f}   "
          f"min {pair_cos.min():.3f}")
    print("   1.00 would mean the withheld identity changed nothing.")

    def spread(E):
        idx = np.random.default_rng(0).choice(len(E), size=min(len(E), 400), replace=False)
        S = E[idx] @ E[idx].T
        iu = np.triu_indices(len(idx), 1)
        return S[iu].mean()
    print("\n4. discriminability (mean pairwise cosine within an arm; lower = more separated)")
    print(f"   full   {spread(ef):.3f}")
    print(f"   masked {spread(em):.3f}")
    print("   If masked is much higher, metadata alone cannot separate the catalogue.")


if __name__ == "__main__":
    main()
