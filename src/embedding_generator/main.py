#!/usr/bin/env python3
"""
main.py — Movie Embedding Generator

Generates multiple embedding types from LLM movie profiles for use as
item features in recommendation models.

Outputs (saved to output/ or --output-dir):
  1. profile_embeddings.npy     — profile text → sentence-transformer → optional PCA
  2. mood_vectors.npy           — direct 10-axis mood feature
  3. theme_matrix.npy           — multi-hot key themes
  4. genome_embeddings.npy      — raw genome tags → PCA (baseline)
  5. combined_features.npy      — profile_emb + mood (recommended)
  6. combined_full.npy          — profile_emb + mood + themes
  7. movie_id_index.json        — ordered movie ID list (row index mapping)
  8. theme_vocabulary.json      — theme names for multi-hot columns
  9. embedding_metadata.json    — model info, dimensions, PCA variance

Usage:
  python main.py                                              # default (bge-large-en-v1.5, no PCA, 1024-dim)
  python main.py --pca-dims 128                               # apply PCA to reduce to 128-dim
  python main.py --pca-dims 256                               # apply PCA to reduce to 256-dim
  python main.py --pca-sweep 128 256 512                      # generate 128, 256, 512, and full-dim variants
  python main.py --model all-MiniLM-L6-v2 --pca-dims 128      # lightweight encoder with PCA
  python main.py --output-dir output/custom                   # custom output directory
  python main.py --skip-genome                                # skip genome baseline
"""

import sys
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    OUTPUT_DIR, PROFILES_JSON,
    DEFAULT_MODEL, PCA_DIMS, GENOME_PCA_DIMS,
)
from embedder import (
    load_profiles,
    encode_profiles,
    extract_mood_vectors,
    encode_key_themes,
    load_genome_vectors,
    build_combined_features,
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def save_outputs(
    output_dir: Path,
    movie_ids,
    profile_emb,
    mood_vectors,
    theme_matrix,
    theme_vocab,
    genome_emb,
    combined,
    combined_full,
    metadata,
):
    """Save all embeddings and metadata to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "profile_embeddings.npy", profile_emb)
    np.save(output_dir / "mood_vectors.npy", mood_vectors)
    np.save(output_dir / "theme_matrix.npy", theme_matrix)
    np.save(output_dir / "combined_features.npy", combined)
    np.save(output_dir / "combined_full.npy", combined_full)

    if genome_emb is not None:
        np.save(output_dir / "genome_embeddings.npy", genome_emb)

    with open(output_dir / "movie_id_index.json", "w") as f:
        json.dump(movie_ids, f)

    with open(output_dir / "theme_vocabulary.json", "w") as f:
        json.dump(theme_vocab, f, ensure_ascii=False)

    with open(output_dir / "embedding_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def save_pca_variant(
    output_dir: Path,
    raw_embeddings: np.ndarray,
    pca_dims: int,
    mood_vectors: np.ndarray,
    theme_matrix: np.ndarray,
    movie_ids,
    theme_vocab,
    genome_emb,
    metadata_base: dict,
    logger,
):
    """Save a PCA-reduced variant of profile embeddings to a subdirectory."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    subdir = output_dir / f"pca{pca_dims}"
    subdir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  PCA sweep: {raw_embeddings.shape[1]} → {pca_dims} dims → {subdir}")
    pca_model = PCA(n_components=pca_dims, random_state=42)
    reduced = pca_model.fit_transform(raw_embeddings)
    reduced = normalize(reduced)
    variance = float(sum(pca_model.explained_variance_ratio_))
    logger.info(f"    Explained variance: {variance:.4f}")

    combined = build_combined_features(reduced, mood_vectors)
    combined_full = build_combined_features(reduced, mood_vectors, theme_matrix)

    metadata = {
        **metadata_base,
        "profile_embedding_dim": pca_dims,
        "profile_pca_applied": True,
        "profile_pca_dims": pca_dims,
        "profile_pca_variance": variance,
        "combined_dim": int(combined.shape[1]),
        "combined_full_dim": int(combined_full.shape[1]),
        "feature_layout": {
            "combined_features": f"profile_emb({pca_dims}) + mood({mood_vectors.shape[1]})",
            "combined_full": f"profile_emb({pca_dims}) + mood({mood_vectors.shape[1]}) + themes({theme_matrix.shape[1]})",
        },
    }

    save_outputs(
        subdir, movie_ids,
        reduced, mood_vectors, theme_matrix, theme_vocab,
        genome_emb, combined, combined_full, metadata,
    )
    logger.info(f"    Saved: profile_embeddings.npy {reduced.shape}")


def main():
    from config import PROFILES_JSON

    parser = argparse.ArgumentParser(description="Generate movie embeddings from LLM profiles")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Sentence-transformer model (default: {DEFAULT_MODEL})")
    parser.add_argument("--pca-dims", type=int, default=PCA_DIMS,
                        help="PCA target dims for profile embeddings (default: None = no PCA)")
    parser.add_argument("--no-pca", action="store_true", help="Skip PCA, keep full dimensions (already the default)")
    parser.add_argument("--pca-sweep", type=int, nargs="+", metavar="DIM",
                        help="Generate multiple PCA variants (e.g., --pca-sweep 128 256 512). "
                             "Also saves the full-dim (no PCA) version.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Custom output directory (default: output/)")
    parser.add_argument("--skip-genome", action="store_true", help="Skip genome baseline embeddings")
    parser.add_argument("--profiles", type=str, default=str(PROFILES_JSON),
                        help="Path to movie_profiles.json")
    args = parser.parse_args()

    # Checked AFTER parsing, and against the path the reader actually chose. Above
    # the parser it did two wrong things: `--help` could not print (the one command
    # someone runs to find out what the flags are), and `--profiles other.json` was
    # rejected on the default path it was passed in order to replace.
    profiles_path = Path(args.profiles)
    if not profiles_path.exists():
        raise SystemExit(
            "Generated profiles not found. They are published in the data release, not\n"
            "in this repository. Download them, or produce them yourself with\n"
            "    python3 src/profile_generator/batch_generate.py\n"
            "or point this script at your own copy with --profiles PATH.\n"
            f"expected at: {profiles_path}")

    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Movie Embedding Generator")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    # In sweep mode, always encode at full dims (PCA applied later per variant)
    if args.pca_sweep:
        pca_dims = None  # encode at full dims
    else:
        pca_dims = None if args.no_pca else args.pca_dims

    # ── Step 1: Load profiles ──
    logger.info("STEP 1: Loading profiles...")
    movie_ids, profiles = load_profiles(Path(args.profiles))

    # ── Step 2: Profile text embeddings ──
    logger.info(f"STEP 2: Encoding profile text with {args.model}...")
    profile_emb, profile_pca = encode_profiles(
        movie_ids, profiles,
        model_name=args.model,
        pca_dims=pca_dims,
    )
    raw_dim = profile_emb.shape[1]

    # ── Step 3: Mood vectors ──
    logger.info("STEP 3: Extracting mood vectors...")
    mood_vectors = extract_mood_vectors(movie_ids, profiles)

    # ── Step 4: Key themes ──
    logger.info("STEP 4: Encoding key themes...")
    theme_matrix, theme_vocab = encode_key_themes(movie_ids, profiles)

    # ── Step 5: Genome baseline ──
    genome_emb = None
    genome_pca = None
    if not args.skip_genome:
        logger.info("STEP 5: Building genome tag baseline...")
        genome_emb, genome_pca = load_genome_vectors(
            movie_ids,
            pca_dims=GENOME_PCA_DIMS,
        )

    # ── Step 6: Combined features ──
    logger.info("STEP 6: Building combined features...")
    combined = build_combined_features(profile_emb, mood_vectors)
    combined_full = build_combined_features(profile_emb, mood_vectors, theme_matrix)

    # ── Step 7: Save ──
    logger.info("STEP 7: Saving outputs...")

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "n_movies": len(movie_ids),
        "sentence_model": args.model,
        "profile_embedding_dim": int(profile_emb.shape[1]),
        "profile_pca_applied": pca_dims is not None,
        "profile_pca_dims": pca_dims,
        "profile_pca_variance": float(sum(profile_pca.explained_variance_ratio_)) if profile_pca else None,
        "mood_vector_dim": int(mood_vectors.shape[1]),
        "theme_vocab_size": len(theme_vocab),
        "genome_embedding_dim": int(genome_emb.shape[1]) if genome_emb is not None else None,
        "genome_pca_dims": GENOME_PCA_DIMS if genome_emb is not None else None,
        "genome_pca_variance": float(sum(genome_pca.explained_variance_ratio_)) if genome_pca else None,
        "combined_dim": int(combined.shape[1]),
        "combined_full_dim": int(combined_full.shape[1]),
        "feature_layout": {
            "combined_features": f"profile_emb({profile_emb.shape[1]}) + mood({mood_vectors.shape[1]})",
            "combined_full": f"profile_emb({profile_emb.shape[1]}) + mood({mood_vectors.shape[1]}) + themes({theme_matrix.shape[1]})",
        },
    }

    save_outputs(
        output_dir, movie_ids,
        profile_emb, mood_vectors, theme_matrix, theme_vocab,
        genome_emb, combined, combined_full, metadata,
    )

    # ── Step 8: PCA sweep (optional) ──
    if args.pca_sweep:
        logger.info(f"STEP 8: PCA dimension sweep: {args.pca_sweep}")
        metadata_base = {
            "generated_at": datetime.now().isoformat(),
            "n_movies": len(movie_ids),
            "sentence_model": args.model,
            "mood_vector_dim": int(mood_vectors.shape[1]),
            "theme_vocab_size": len(theme_vocab),
            "genome_embedding_dim": int(genome_emb.shape[1]) if genome_emb is not None else None,
            "genome_pca_dims": GENOME_PCA_DIMS if genome_emb is not None else None,
            "genome_pca_variance": float(sum(genome_pca.explained_variance_ratio_)) if genome_pca else None,
        }
        for dim in args.pca_sweep:
            if dim >= raw_dim:
                logger.info(f"  Skipping PCA-{dim}: native dim is {raw_dim}")
                continue
            save_pca_variant(
                output_dir, profile_emb, dim,
                mood_vectors, theme_matrix,
                movie_ids, theme_vocab, genome_emb,
                metadata_base, logger,
            )

    # ── Summary ──
    logger.info("=" * 60)
    logger.info("DONE. Output files:")
    logger.info(f"  profile_embeddings.npy   {profile_emb.shape}")
    logger.info(f"  mood_vectors.npy         {mood_vectors.shape}")
    logger.info(f"  theme_matrix.npy         {theme_matrix.shape}")
    if genome_emb is not None:
        logger.info(f"  genome_embeddings.npy    {genome_emb.shape}")
    logger.info(f"  combined_features.npy    {combined.shape}")
    logger.info(f"  combined_full.npy        {combined_full.shape}")
    logger.info(f"  movie_id_index.json      {len(movie_ids)} IDs")
    logger.info(f"  theme_vocabulary.json     {len(theme_vocab)} themes")
    logger.info(f"  embedding_metadata.json")
    logger.info(f"  Output dir: {output_dir}")
    if args.pca_sweep:
        logger.info(f"  PCA variants in: {[f'pca{d}' for d in args.pca_sweep if d < raw_dim]}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
