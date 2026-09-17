"""The released profiles and embeddings are what the paper says they are.

Every assertion here is against the RELEASED distribution, not the prompt's
specification. The previous version asserted the spec and failed on the artifact:
"exactly 3 key themes" where 36% of records carry more, mood values in [0, 1]
where the axes run [-1, 1], and a `profile_text` field that is named `profile`.
That is the same defect the datasheet carried, sitting in the one place designed
to catch it -- so these tests now check the realised distribution and say what it
is, and a change to the artifact has to come past them.
"""
from __future__ import annotations

import numpy as np
import pytest

CATALOGUE = 10381
MOOD_AXES = 10
THEME_VOCAB = 528


class TestProfileData:
    def test_profile_count(self, profiles):
        assert len(profiles) == CATALOGUE

    def test_profile_fields(self, profiles):
        required = {"movieId", "title", "profile", "word_count", "mood_vector",
                    "key_themes"}
        missing = {k for r in profiles.values() for k in required - set(r)}
        assert not missing, f"fields absent from at least one record: {missing}"

    def test_mood_vector_is_named_axes(self, profiles):
        """A dict keyed by axis, not a bare sequence.

        The order of these keys is what row i of mood_vectors.npy follows, so the
        shape of this field is load-bearing and not cosmetic.
        """
        sample = next(iter(profiles.values()))["mood_vector"]
        assert isinstance(sample, dict) and len(sample) == MOOD_AXES
        keys = {tuple(r["mood_vector"]) for r in profiles.values()}
        assert len(keys) == 1, "records disagree on the mood axis order"

    def test_mood_values_in_range(self, profiles):
        """[-1, 1], as the paper states. The old bound of [0, 1] was wrong."""
        bad = [(m, a, v) for m, r in profiles.items()
               for a, v in r["mood_vector"].items() if not -1.0 <= float(v) <= 1.0]
        assert not bad, f"{len(bad)} mood value(s) outside [-1, 1], e.g. {bad[:3]}"

    def test_key_themes_realised_range(self, profiles):
        """3-5 requested; 3-6 realised, four records carrying six."""
        counts = [len(r["key_themes"]) for r in profiles.values()]
        assert min(counts) == 3 and max(counts) == 6
        assert sum(1 for c in counts if c == 6) == 4
        assert sum(1 for c in counts if c > 3) == 3693

    def test_profile_word_count_realised_range(self, profiles):
        """95-135 realised against 80-120 requested, median 115, 20.2% over 120."""
        words = [len(r["profile"].split()) for r in profiles.values()]
        assert min(words) == 95 and max(words) == 135
        over = sum(1 for w in words if w > 120)
        assert 0.20 <= over / len(words) <= 0.21

    def test_word_count_field_agrees_with_the_text(self, profiles):
        """The generator overrides the model's self-reported count; check it did."""
        off = [m for m, r in profiles.items()
               if abs(int(r["word_count"]) - len(r["profile"].split())) > 1]
        assert not off, f"{len(off)} record(s) whose word_count disagrees with profile"


class TestEmbeddings:
    def _npy(self, d, name):
        p = d / name
        if not p.exists():
            pytest.skip(f"{name} not present")
        return np.load(p, mmap_mode="r")

    def test_profile_embeddings_shape(self, embeddings_dir):
        assert self._npy(embeddings_dir, "profile_embeddings.npy").shape == (CATALOGUE, 1024)

    def test_mood_vectors_shape(self, embeddings_dir):
        assert self._npy(embeddings_dir, "mood_vectors.npy").shape == (CATALOGUE, MOOD_AXES)

    def test_theme_matrix_shape(self, embeddings_dir):
        assert self._npy(embeddings_dir, "theme_matrix.npy").shape == (CATALOGUE, THEME_VOCAB)

    def test_combined_features_is_profile_and_mood(self, embeddings_dir):
        cf = self._npy(embeddings_dir, "combined_features.npy")
        pe = self._npy(embeddings_dir, "profile_embeddings.npy")
        mv = self._npy(embeddings_dir, "mood_vectors.npy")
        assert cf.shape == (CATALOGUE, 1034)
        assert np.allclose(cf[7], np.concatenate([pe[7], mv[7]]), atol=1e-5)

    def test_combined_full_adds_themes(self, embeddings_dir):
        cfull = self._npy(embeddings_dir, "combined_full.npy")
        parts = [self._npy(embeddings_dir, n) for n in
                 ("profile_embeddings.npy", "mood_vectors.npy", "theme_matrix.npy")]
        assert cfull.shape == (CATALOGUE, 1562)
        assert np.allclose(cfull[7], np.concatenate([p[7] for p in parts]), atol=1e-5)

    def test_no_nan(self, embeddings_dir):
        for n in ("profile_embeddings.npy", "mood_vectors.npy", "theme_matrix.npy"):
            assert not np.isnan(self._npy(embeddings_dir, n)).any(), f"NaN in {n}"


class TestRowAlignment:
    """Row i of every array is the item movie_id_index.json names at i.

    Zipping the profile JSON to the array rows instead returns the wrong item for
    199 of 200 sampled rows: the two hold the same ids in different orders. This
    is the artifact's central join and the reason the index ships.
    """

    def test_index_length(self, movie_id_index):
        assert len(movie_id_index) == CATALOGUE

    def test_index_and_profiles_name_the_same_items(self, movie_id_index, profiles):
        assert {str(m) for m in movie_id_index} == set(profiles)

    def test_rows_follow_the_index(self, movie_id_index, profiles, embeddings_dir):
        mv = np.load(embeddings_dir / "mood_vectors.npy")
        axes = list(next(iter(profiles.values()))["mood_vector"])
        want = np.array([[profiles[str(m)]["mood_vector"][a] for a in axes]
                         for m in movie_id_index], dtype=np.float32)
        assert np.isclose(np.asarray(mv, dtype=np.float32), want, atol=1e-4).all()
