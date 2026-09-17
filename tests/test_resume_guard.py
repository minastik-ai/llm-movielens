"""Tests for the resume guard: an interrupted run must be resumable, a finished
one must not be silently continued.

Two requirements, both about `training_state.pt`:

  R1  A run killed mid-flight resumes at the next epoch, never from scratch.
  R2  A run that FINISHED — by early stopping OR by exhausting its epoch budget
      — refuses to continue, loudly, so `best_model.pt` cannot be overwritten
      and an already-published number cannot silently change.

R2's second half is the subtle one. Early stopping was covered; finishing a full
200/200 run was not. Re-running such a run with a larger `--epochs` used to walk
straight past the guard, because "finished" was defined as "stopped early".
"""

from pathlib import Path


import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _layout import component, load_module
# Was `parents[1] / "code" / "benchmark"`, which exists only in the development
# checkout -- the release flattens it to src/, so this module could not even be
# COLLECTED there and took the whole suite down with it. The loader resolves both
# layouts AND turns train.py's own missing imports into a skip rather than the
# collection error that stopped the entire suite on a torch-free interpreter.


BENCH = component("benchmark")
train = load_module("benchmark", "train.py")


PATIENCE = 20


def state(epoch, best_epoch, patience_counter, stopped_early=None, target_epochs=None):
    """A minimal training state — only the fields the guard reads."""
    s = {"epoch": epoch, "best_epoch": best_epoch, "patience_counter": patience_counter}
    if stopped_early is not None:
        s["stopped_early"] = stopped_early
    if target_epochs is not None:
        s["target_epochs"] = target_epochs
    return s


class TestInterruptedRunIsResumable:
    """R1 — killed mid-run must resume, not restart and not refuse."""

    def test_killed_midrun_is_not_finished(self):
        # epoch 6 of 200, patience_counter 5 of 20 — clearly still improving
        s = state(epoch=6, best_epoch=1, patience_counter=5, target_epochs=200)
        finished, _ = train._run_is_finished(s, PATIENCE, requested_epochs=200)
        assert finished is False

    def test_legacy_killed_midrun_is_not_finished(self):
        """No stopped_early, no target_epochs — the pre-fix format."""
        s = state(epoch=6, best_epoch=1, patience_counter=5)
        finished, _ = train._run_is_finished(s, PATIENCE, requested_epochs=200)
        assert finished is False

    def test_resume_starts_at_next_epoch_not_scratch(self):
        s = state(epoch=57, best_epoch=50, patience_counter=5, target_epochs=200)
        finished, _ = train._run_is_finished(s, PATIENCE, requested_epochs=200)
        assert finished is False
        assert s["epoch"] + 1 == 58, "resume must continue at 58, not 1"


class TestEarlyStoppedRunIsProtected:
    """R2a — early stopping already recorded."""

    def test_explicit_flag(self):
        s = state(epoch=105, best_epoch=85, patience_counter=20,
                  stopped_early=True, target_epochs=200)
        finished, why = train._run_is_finished(s, PATIENCE, requested_epochs=200)
        assert finished is True
        assert "early" in why.lower()

    def test_derived_for_legacy_checkpoint(self):
        """Pre-field checkpoints: patience_counter >= patience IS the break condition."""
        s = state(epoch=105, best_epoch=85, patience_counter=20)
        finished, why = train._run_is_finished(s, PATIENCE, requested_epochs=200)
        assert finished is True
        assert "early" in why.lower()

    def test_still_protected_when_more_epochs_requested(self):
        """The dangerous case: --epochs 300 against an early-stopped run."""
        s = state(epoch=105, best_epoch=85, patience_counter=20,
                  stopped_early=True, target_epochs=200)
        finished, _ = train._run_is_finished(s, PATIENCE, requested_epochs=300)
        assert finished is True


class TestBudgetExhaustedRunIsProtected:
    """R2b — the gap: a run that finished by reaching its epoch budget.

    stopped_early is False (it never early-stopped), so an early-stop-only
    predicate says "not finished" and happily trains on.
    """

    def test_full_run_with_larger_epochs_is_still_finished(self):
        s = state(epoch=200, best_epoch=180, patience_counter=5,
                  stopped_early=False, target_epochs=200)
        finished, why = train._run_is_finished(s, PATIENCE, requested_epochs=300)
        assert finished is True, "a completed 200/200 run must not silently continue to 300"
        assert "budget" in why.lower() or "200" in why

    def test_full_run_at_same_epochs_is_finished(self):
        s = state(epoch=200, best_epoch=180, patience_counter=5,
                  stopped_early=False, target_epochs=200)
        finished, _ = train._run_is_finished(s, PATIENCE, requested_epochs=200)
        assert finished is True

    def test_legacy_full_run_falls_back_to_requested(self):
        """No target_epochs recorded; epoch >= what is being asked for."""
        s = state(epoch=200, best_epoch=180, patience_counter=5)
        finished, _ = train._run_is_finished(s, PATIENCE, requested_epochs=200)
        assert finished is True

    def test_genuine_extension_of_a_shorter_budget_is_allowed(self):
        """Ran 50/50 deliberately, now extending to 200 — legitimate, allow it.

        Distinguishable ONLY because target_epochs was recorded: the run met a
        50-epoch budget, and the user is now asking for a larger one.
        """
        s = state(epoch=50, best_epoch=45, patience_counter=5, target_epochs=50)
        finished, _ = train._run_is_finished(s, PATIENCE, requested_epochs=200)
        assert finished is True, (
            "reaching its budget still counts as finished — continuing must be "
            "an explicit choice, since it can overwrite best_model.pt"
        )


class TestGuardIsLoud:
    """R2 — 'a warning is raised'. INFO scrolls past; WARNING does not."""

    def test_refusal_is_logged_at_warning_level(self):
        """Locate the guard's own logging call via the AST, not a text window.

        A source-substring search finds the *definition* of `_run_is_finished`
        before its call site, so it proves nothing. Walk the tree instead and
        require that the message naming the refusal is a `logger.warning(...)`.
        """
        import ast

        tree = ast.parse((BENCH / "train.py").read_text())
        levels = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Attribute) and getattr(fn.value, "id", None) == "logger"):
                continue
            text = " ".join(
                seg.value for arg in node.args for seg in ast.walk(arg)
                if isinstance(seg, ast.Constant) and isinstance(seg.value, str)
            )
            if "REFUSING TO CONTINUE" in text:
                levels.add(fn.attr)
        assert levels, "no log call carries the refusal message"
        assert levels == {"warning"}, f"refusal must be WARNING, found {levels}"


class TestBestCheckpointBackup:
    """R2 — 'prevent the best checkpoint from being overwritten'."""

    def test_forced_continuation_backs_up_best_model(self, tmp_path):
        best = tmp_path / "best_model.pt"
        best.write_bytes(b"original-weights")
        backup = train._backup_best_model(tmp_path)
        assert backup is not None and backup.exists()
        assert backup.read_bytes() == b"original-weights"
        assert best.read_bytes() == b"original-weights", "backup must not move the original"

    def test_backup_is_noop_when_no_best_model(self, tmp_path):
        assert train._backup_best_model(tmp_path) is None

    def test_backup_does_not_clobber_an_existing_backup(self, tmp_path):
        best = tmp_path / "best_model.pt"
        best.write_bytes(b"v1")
        first = train._backup_best_model(tmp_path)
        best.write_bytes(b"v2")
        second = train._backup_best_model(tmp_path)
        assert first.read_bytes() == b"v1", "the first backup is the published one; keep it"
        assert second != first


class TestPatienceIsRecordedAndHonoured:
    """The early-stop derivation compares patience_counter against a patience.

    Which patience? The run's OWN. `patience` is configurable per researcher, so
    reading today's config value against a checkpoint trained under a different
    one silently changes the verdict: a run that early-stopped at patience 5
    looks 'still improving' when re-read under patience 20.
    """

    def test_state_records_patience(self):
        src = (BENCH / "train.py").read_text()
        assert '"patience": patience' in src, "training state must persist the run's patience"

    def test_runs_own_patience_wins_over_current_config(self):
        # trained with patience=5 and early-stopped; re-read under today's 20
        s = state(epoch=40, best_epoch=35, patience_counter=5)
        s["patience"] = 5
        finished, why = train._run_is_finished(s, PATIENCE, requested_epochs=200)
        assert finished is True, "must use the run's own patience (5), not the current 20"
        assert "early" in why.lower()

    def test_falls_back_to_current_patience_when_absent(self):
        s = state(epoch=105, best_epoch=85, patience_counter=20)  # no 'patience' key
        finished, _ = train._run_is_finished(s, PATIENCE, requested_epochs=200)
        assert finished is True


class TestBestModelCarriesProvenance:
    """`best_model.pt` is a bare state_dict that external tools load directly.

    Its format must NOT change — the released HF spot-check and the eval scripts
    all do `load_state_dict(torch.load(best_model.pt))`. So provenance goes in a
    sidecar next to it.
    """

    def test_sidecar_written_with_budget_and_stop_status(self, tmp_path):
        train._write_best_model_meta(
            tmp_path, epoch=105, best_epoch=85, best_ndcg=0.1169,
            target_epochs=200, patience=20, stopped_early=True,
        )
        import json
        meta = json.loads((tmp_path / "best_model.meta.json").read_text())
        assert meta["target_epochs"] == 200
        assert meta["patience"] == 20
        assert meta["stopped_early"] is True
        assert meta["best_epoch"] == 85

    def test_sidecar_does_not_touch_the_checkpoint_format(self, tmp_path):
        best = tmp_path / "best_model.pt"
        best.write_bytes(b"weights")
        train._write_best_model_meta(tmp_path, epoch=1, best_epoch=1, best_ndcg=0.0,
                                     target_epochs=200, patience=20, stopped_early=False)
        assert best.read_bytes() == b"weights", "best_model.pt must remain a bare state_dict"
