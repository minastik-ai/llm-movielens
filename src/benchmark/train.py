"""
BPR training loop with validation, early stopping, checkpointing, and resume support.

Resume behavior:
  - INTERRUPTED run (killed, preempted, crashed): training_state.pt is written
    atomically after EVERY epoch, so resume continues at the next epoch — never
    from scratch. Model, optimizer, best-so-far, patience counter and all RNG
    streams are restored, so a resumed run reproduces an uninterrupted one.
  - FINISHED run (early-stopped, or reached its epoch budget): resume REFUSES
    and logs a warning. Continuing could find a new best, overwrite
    best_model.pt and silently change an already-published number. Override with
    --ignore-early-stop, which backs up best_model.pt first.
  - Training state is ALWAYS preserved (never deleted) to allow extending
    training as a deliberate, flagged act.
"""

import time
import json
import random
import shutil
import logging
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE, BATCH_SIZE,
    CHECKPOINT_DIR, RESULTS_DIR, TOP_K,
)
from data.dataset import InteractionData, get_train_loader
from evaluate import evaluate_model

logger = logging.getLogger(__name__)


def bpr_loss(pos_scores, neg_scores):
    """BPR loss: -log(sigmoid(pos - neg))."""
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def train_epoch(model, loader, optimizer, weight_decay, device):
    """Train one epoch with BPR loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for users, pos_items, neg_items in loader:
        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)

        pos_scores, neg_scores, reg_loss = model(users, pos_items, neg_items)
        loss = bpr_loss(pos_scores, neg_scores) + weight_decay * reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def _capture_rng(loader_generator=None):
    """Snapshot every RNG stream that influences training.

    Without this, a resumed run re-seeds all RNGs from `seed` and therefore
    draws a DIFFERENT negative-sampling / shuffling stream than an
    uninterrupted run — so `--seed 42 (resumed)` would not reproduce
    `--seed 42 (straight through)`. Streams captured:
      torch (model init / dropout), numpy (BPRDataset negative sampling via
      the GLOBAL np RNG in __getitem__), python random, and the DataLoader
      shuffle generator.
    """
    rng = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "loader_generator": loader_generator.get_state() if loader_generator is not None else None,
    }
    if torch.cuda.is_available():
        rng["torch_cuda"] = torch.cuda.get_rng_state_all()
    return rng


def _as_byte_cpu(t):
    """torch RNG states must be CPU uint8 tensors.

    torch.load(map_location=<device>) moves saved tensors onto the training
    device, so an MPS/CUDA-resident state must be forced back to CPU uint8
    before set_rng_state() will accept it.
    """
    if t is None:
        return None
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.uint8)
    return t.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def _restore_rng(rng):
    """Restore RNG streams saved by _capture_rng. Tolerant of older checkpoints."""
    if not rng:
        logger.warning("  Checkpoint has no RNG state (pre-fix run): resumed stream "
                       "will DIFFER from an uninterrupted run with the same seed.")
        return None
    torch.set_rng_state(_as_byte_cpu(rng["torch"]))
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])
    if rng.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_as_byte_cpu(x) for x in rng["torch_cuda"]])
    return _as_byte_cpu(rng.get("loader_generator"))


def _save_training_state(checkpoint_dir, epoch, model, optimizer,
                         best_ndcg, best_metrics, best_epoch, patience_counter, history,
                         loader_generator=None, stopped_early=False, target_epochs=None,
                         patience=None):
    """Save full training state for resume (atomic: tmp file + rename).

    `stopped_early` records WHY the run ended. Without it a resume cannot tell
    "stopped at epoch 105 because early stopping fired" from "stopped at epoch
    105 because the process was killed", so it resumes and trains on toward
    --epochs -- which can find a new best and overwrite an already-published
    result. See _resume_is_complete().
    """
    tmp = checkpoint_dir / "training_state.pt.tmp"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_ndcg": best_ndcg,
        "best_metrics": best_metrics,
        "best_epoch": best_epoch,
        "patience_counter": patience_counter,
        "history": history,
        "stopped_early": stopped_early,
        "target_epochs": target_epochs,
        "patience": patience,
        "rng": _capture_rng(loader_generator),
    }, tmp)
    tmp.replace(checkpoint_dir / "training_state.pt")


def _run_is_finished(state, patience, requested_epochs):
    """Did the run this state came from FINISH, or was it interrupted?

    Returns ``(finished, reason)``. Two ways to finish, and both must be caught
    — a finished run that gets resumed can find a new "best", overwrite
    ``best_model.pt``, and silently change an already-published number:

      1. Early stopping. See `_resume_is_complete()`.
      2. Exhausting its epoch budget. `stopped_early` is False for these, so an
         early-stop-only check calls them unfinished and trains on. Re-running a
         completed 200/200 run with `--epochs 300` used to walk straight past
         the guard.

    `target_epochs` is what makes (2) decidable: it records the budget the run
    was launched with, so "ran 200 of 200" is distinguishable from "killed at
    200 of 500". Checkpoints written before the field existed fall back to
    comparing against what is being requested now — the same arithmetic the
    epoch loop already applied, just made explicit and loud.

    Note that legitimately extending a finished short run (50/50, now asking for
    200) also reports finished. That is deliberate: it is a real continuation
    that can overwrite the best checkpoint, so it should be an explicit choice
    rather than a silent side effect.
    """
    # Prefer the patience the run was TRAINED with. `patience` is configurable
    # per researcher, so judging an old checkpoint by today's value silently
    # changes the verdict: a run that early-stopped under patience 5 reads as
    # "still improving" under patience 20, and would be resumed.
    eff_patience = state.get("patience") or patience
    if _resume_is_complete(state, eff_patience):
        return True, f"early stopping (patience {eff_patience})"

    epoch = state.get("epoch", 0)
    target = state.get("target_epochs")
    if target is not None:
        if epoch >= target:
            return True, f"completed its full {target}-epoch budget"
        return False, None

    if epoch >= requested_epochs:
        return True, f"already reached epoch {epoch} of the {requested_epochs} requested"
    return False, None


def _write_best_model_meta(checkpoint_dir, epoch, best_epoch, best_ndcg,
                           target_epochs, patience, stopped_early):
    """Record provenance for `best_model.pt` in a SIDECAR, not inside the file.

    `best_model.pt` is a bare `state_dict`, and external consumers load it
    directly -- the eval scripts all do
    `model.load_state_dict(torch.load(best_model.pt))`. Wrapping the weights in
    a metadata dict would break every one of them, including a published
    artifact. So the provenance goes next to the file instead:

        best_model.pt            <- unchanged, still a bare state_dict
        best_model.meta.json     <- which budget produced it, and how it ended

    That is what lets a later reader (or a later YOU) see that these weights came
    from a run that stopped early at epoch 105 under patience 20 of a 200-epoch
    budget -- without opening a 200 MB checkpoint or guessing from filenames.
    """
    meta = {
        "best_epoch": best_epoch,
        "final_epoch": epoch,
        "best_ndcg@10": best_ndcg,
        "target_epochs": target_epochs,
        "patience": patience,
        "stopped_early": stopped_early,
        "note": "best_model.pt is a bare state_dict; this sidecar carries its provenance.",
    }
    path = Path(checkpoint_dir) / "best_model.meta.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    tmp.replace(path)
    return path


def _backup_best_model(checkpoint_dir):
    """Copy `best_model.pt` aside before a forced continuation can overwrite it.

    Returns the backup path, or None if there is no best model yet. Never
    overwrites an existing backup — the FIRST one is the published weight set,
    and that is the one worth keeping — so repeated continuations accumulate
    `.pre-continue`, `.pre-continue.2`, ... rather than destroying the original.
    """
    best = Path(checkpoint_dir) / "best_model.pt"
    if not best.exists():
        return None
    backup = best.with_suffix(".pt.pre-continue")
    n = 2
    while backup.exists():
        backup = best.with_suffix(f".pt.pre-continue.{n}")
        n += 1
    shutil.copy2(best, backup)
    return backup


def _resume_is_complete(state, patience):
    """Did the run this state came from finish, or was it interrupted?

    Checkpoints written before the `stopped_early` field existed do not say. For
    those, derive it: `patience_counter >= patience` is exactly the early-stop
    break condition, so a state carrying that condition can only have come from
    a run that stopped early. This recovers the correct answer for pre-existing
    checkpoints without re-running them.
    """
    flag = state.get("stopped_early")
    if flag is not None:
        return bool(flag)
    return state.get("patience_counter", 0) >= patience


def _load_training_state(checkpoint_dir, model, optimizer, device):
    """Load training state for resume. Returns None if no checkpoint exists."""
    state_path = checkpoint_dir / "training_state.pt"
    if not state_path.exists():
        return None

    state = torch.load(state_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    state["_loader_generator_state"] = _restore_rng(state.get("rng"))

    logger.info(f"  Resumed from epoch {state['epoch']} "
                f"(best NDCG@10: {state['best_ndcg']:.4f} at epoch {state['best_epoch']})")
    return state


def train_model(
    model,
    interaction_data: InteractionData,
    device: str = "cpu",
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    num_epochs: int = NUM_EPOCHS,
    patience: int = PATIENCE,
    batch_size: int = BATCH_SIZE,
    eval_every: int = 5,
    experiment_name: str = "default",
    resume: bool = True,
    ignore_early_stop: bool = False,
    checkpoint_dir: Path = None,
    results_dir: Path = None,
    seed: int = 42,
) -> Dict:
    """
    Full training loop with validation, early stopping, and resume support.

    Resume behavior:
      - If training_state.pt exists and epoch < num_epochs: continue training
      - If training_state.pt exists and epoch >= num_epochs: re-evaluate and save results
      - Training state is always preserved to allow extending with more epochs
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    # Own the shuffle generator so its state can be saved/restored on resume.
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    loader = get_train_loader(interaction_data, batch_size=batch_size, seed=seed,
                              generator=loader_generator)

    best_ndcg = 0.0
    best_metrics = {}
    best_epoch = 0
    patience_counter = 0
    history = []
    start_epoch = 1

    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR
    if results_dir is None:
        results_dir = RESULTS_DIR
    exp_checkpoint_dir = checkpoint_dir / experiment_name
    exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Try to resume from checkpoint
    if resume:
        state = _load_training_state(exp_checkpoint_dir, model, optimizer, device)
        if state is not None:
            start_epoch = state["epoch"] + 1
            best_ndcg = state["best_ndcg"]
            best_metrics = state["best_metrics"]
            best_epoch = state["best_epoch"]
            patience_counter = state["patience_counter"]
            history = state["history"]
            _gen_state = state.get("_loader_generator_state")
            if _gen_state is not None:
                loader_generator.set_state(_gen_state)

            finished, why = _run_is_finished(state, patience, num_epochs)
            if finished and not ignore_early_stop:
                logger.warning(
                    f"  REFUSING TO CONTINUE {experiment_name}: this run already FINISHED "
                    f"at epoch {state['epoch']} — {why} (best epoch {best_epoch}, "
                    f"NDCG@10 {best_ndcg:.4f}). Training on toward epoch {num_epochs} could "
                    f"find a new best, overwrite best_model.pt, and silently change a "
                    f"published result. Re-running test evaluation only. "
                    f"Pass --ignore-early-stop to continue anyway (best_model.pt will be "
                    f"backed up first)."
                )
                start_epoch = num_epochs + 1
            elif finished and ignore_early_stop:
                backup = _backup_best_model(exp_checkpoint_dir)
                logger.warning(
                    f"  CONTINUING a finished run ({why}) because --ignore-early-stop was "
                    f"passed. best_model.pt backed up to {backup.name if backup else '(none yet)'}."
                )

    if start_epoch > num_epochs:
        logger.info(f"No training left for {experiment_name} — running test evaluation only")
    else:
        logger.info(f"Training {experiment_name}: epochs {start_epoch}-{num_epochs}, lr={lr}, wd={weight_decay}")

    for epoch in range(start_epoch, num_epochs + 1):
        # Log epoch start
        epoch_start_time = datetime.now().strftime("%H:%M:%S")
        logger.info(f"[{epoch_start_time}] Epoch {epoch:3d}/{num_epochs} starting...")

        t0 = time.time()
        train_loss = train_epoch(model, loader, optimizer, weight_decay, device)
        train_time = time.time() - t0

        log_entry = {"epoch": epoch, "train_loss": train_loss, "train_time": train_time}

        # Evaluate on validation set
        if epoch % eval_every == 0 or epoch == 1:
            t0 = time.time()
            val_metrics = evaluate_model(model, interaction_data, split="val", device=device)
            eval_time = time.time() - t0

            ndcg10 = val_metrics.get("NDCG@10", 0)
            log_entry.update(val_metrics)
            log_entry["eval_time"] = eval_time

            epoch_end_time = datetime.now().strftime("%H:%M:%S")
            logger.info(
                f"[{epoch_end_time}] Epoch {epoch:3d}/{num_epochs} done | "
                f"Loss: {train_loss:.4f} | "
                f"NDCG@10: {ndcg10:.4f} | "
                f"Recall@20: {val_metrics.get('Recall@20', 0):.4f} | "
                f"Time: {train_time:.1f}s train + {eval_time:.1f}s eval"
            )

            if ndcg10 > best_ndcg:
                best_ndcg = ndcg10
                best_metrics = val_metrics.copy()
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), exp_checkpoint_dir / "best_model.pt")
                _write_best_model_meta(exp_checkpoint_dir, epoch=epoch, best_epoch=best_epoch,
                                       best_ndcg=best_ndcg, target_epochs=num_epochs,
                                       patience=patience, stopped_early=False)
                logger.info(f"  → New best NDCG@10: {ndcg10:.4f} (saved)")
            else:
                patience_counter += eval_every
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch} (best: {best_epoch})")
                    _write_best_model_meta(exp_checkpoint_dir, epoch=epoch,
                                           best_epoch=best_epoch, best_ndcg=best_ndcg,
                                           target_epochs=num_epochs, patience=patience,
                                           stopped_early=True)
                    history.append(log_entry)
                    # Save state before breaking
                    _save_training_state(exp_checkpoint_dir, epoch, model, optimizer,
                                         best_ndcg, best_metrics, best_epoch, patience_counter,
                                         history, loader_generator, stopped_early=True,
                                         target_epochs=num_epochs, patience=patience)
                    break
        else:
            epoch_end_time = datetime.now().strftime("%H:%M:%S")
            logger.info(
                f"[{epoch_end_time}] Epoch {epoch:3d}/{num_epochs} done | "
                f"Loss: {train_loss:.4f} | "
                f"Time: {train_time:.1f}s train (no eval)"
            )

        history.append(log_entry)

        # Save training state after every epoch for resume
        _save_training_state(exp_checkpoint_dir, epoch, model, optimizer,
                             best_ndcg, best_metrics, best_epoch, patience_counter,
                             history, loader_generator, target_epochs=num_epochs,
                             patience=patience)

    # Load best model and evaluate on test set
    best_model_path = exp_checkpoint_dir / "best_model.pt"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
    logger.info(f"Evaluating on test set (best model from epoch {best_epoch})...")
    test_metrics = evaluate_model(model, interaction_data, split="test", device=device)

    logger.info(f"═══ Test Results ({experiment_name}) ═══")
    for k, v in sorted(test_metrics.items()):
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")

    # Save final results
    results = {
        "experiment": experiment_name,
        "best_epoch": best_epoch,
        "total_epochs_trained": max(e["epoch"] for e in history) if history else 0,
        "stopped_early": patience_counter >= patience,
        "best_val_metrics": best_metrics,
        "test_metrics": test_metrics,
        "config": {
            "lr": lr,
            "weight_decay": weight_decay,
            "num_epochs": num_epochs,
            "patience": patience,
            # The unit is the whole point: this trainer counts patience in EPOCHS,
            # while several widely used implementations count it in EVALUATIONS, so
            # the bare number is ambiguous. The paper recommends
            # recording the unit; recording it here is what makes that true.
            "patience_unit": "epochs",
            "batch_size": batch_size,
        },
    }

    exp_results_dir = results_dir / experiment_name
    exp_results_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Training state is KEPT (not deleted) so training can be extended later
    # To continue training: increase --epochs and re-run
    logger.info(f"  Results saved. Training state preserved for potential continuation.")

    return results
