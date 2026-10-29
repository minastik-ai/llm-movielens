"""The training loader's worker processes.

The loader's worker initialiser is a local function, which cannot be pickled under `forkserver` or `spawn`
(`PicklingError: Can't pickle local object ... _worker_init`), the start methods some platforms default to.
The loader therefore starts its workers with `fork` on Linux, so the negatives are drawn the same way
whatever the interpreter's default. These tests check that the loader works under a non-fork default and
that the draws do not change.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import platform

import numpy as np
import pytest

from _layout import load_module

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")
D = load_module("benchmark", "data/dataset.py")

pytestmark = pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="no fork on this platform")


@pytest.fixture()
def data(tmp_path):
    rng = np.random.default_rng(0)
    nu, ni = 40, 30
    rows = [(u, int(i), 1000 + k) for u in range(nu) for k, i in enumerate(rng.choice(ni, 12, replace=False))]
    pd.DataFrame(rows, columns=["userId", "movieId", "timestamp"]).to_csv(tmp_path / "train.csv", index=False)
    for name in ("val", "test"):
        pd.DataFrame(rows[:5], columns=["userId", "movieId", "timestamp"]).to_csv(tmp_path / f"{name}.csv", index=False)
    (tmp_path / "stats.json").write_text(json.dumps({"n_users": nu, "n_items": ni}))
    (tmp_path / "item_map.json").write_text(json.dumps({str(i): i for i in range(ni)}))
    return D.InteractionData(data_dir=tmp_path)


def _batches(loader, n=3):
    out = []
    for i, b in enumerate(loader):
        out.append([t.clone() for t in b])
        if i + 1 == n:
            break
    return out


def test_linux_workers_start_with_fork_whatever_the_default(monkeypatch, data):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    loader = D.get_train_loader(data, batch_size=64, seed=7, num_workers=2)
    assert loader.multiprocessing_context is not None and loader.multiprocessing_context.get_start_method() == "fork"
    assert len(_batches(loader)) == 3            # under the old loader this raises PicklingError off fork


def test_the_draws_are_those_of_the_fork_loader_the_published_runs_used(monkeypatch, data):
    """A reference loader built the old way, with fork named explicitly and the same seed + worker_id
    initialiser, must give the same batches."""
    import random as _random
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    seed = 7

    def init(worker_id):
        np.random.seed(seed + worker_id)
        _random.seed(seed + worker_id)

    gen = torch.Generator(); gen.manual_seed(seed)
    ref = torch.utils.data.DataLoader(D.BPRDataset(data), batch_size=64, shuffle=True, num_workers=2,
                                      drop_last=True, generator=gen, worker_init_fn=init,
                                      multiprocessing_context="fork")
    mine = D.get_train_loader(data, batch_size=64, seed=seed, num_workers=2)
    for a, b in zip(_batches(ref), _batches(mine)):
        assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_macos_keeps_zero_workers_and_no_context(monkeypatch, data):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    loader = D.get_train_loader(data, batch_size=64, seed=7)
    assert loader.num_workers == 0 and loader.multiprocessing_context is None


def test_no_workers_means_no_context(monkeypatch, data):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    loader = D.get_train_loader(data, batch_size=64, seed=7, num_workers=0)
    assert loader.num_workers == 0 and loader.multiprocessing_context is None
