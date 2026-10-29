#!/usr/bin/env python3
"""Select the contrastive baselines' own settings on validation, then train their reported runs.

The three baselines are M1b SimGCL, M1c XSimGCL and M1d LightGCL.

WHY. None of the three methods was evaluated on MovieLens 20M by its authors, and each
paper tuned its own hyperparameters per dataset (SimGCL's best lambda runs from 0.05 to 2
across the datasets its authors report). Copying another dataset's value is arbitrary:
at lambda = 0.5 (SimGCL's Yelp2018 value), validation NDCG@10 on ML-20M fell at each of
its first three evaluations. So this does what the authors did -- select each method's
own hyperparameters on the validation split, within the ranges they report -- and keeps
every other setting on the harness's shared protocol.

THE PROTOCOL, fixed before any run:
  Search     each method's own hyperparameters that its authors set differently across
             datasets, over values they report, on validation NDCG@10 at seed 42.
               SimGCL    lambda in {0.05, 0.1, 0.2, 0.5}; eps 0.1 and tau 0.2 fixed
               XSimGCL   in the authors' own order: lambda in {0.05, 0.1, 0.2} (l* 1, eps
                         0.1), then l* in {1, 2, 3}, then eps in {0.05, 0.1, 0.2}; tau 0.2
               LightGCL  tau in {0.1, 0.2, 0.5}; lambda_1 0.2 and q 5 fixed
  Edges      a winner at the end of a searched range is extended outward along a ladder
             of values the authors explored, and the stage re-selects: SimGCL lambda
             0.02, 0.01 / 1, 2; XSimGCL lambda 0.02, 0.01 / 0.5, 1; LightGCL tau 1.
  Selection  the run's best validation NDCG@10; exact ties go to the smaller value. A
             selection run never evaluates on test. A model's choice is frozen once
             made, and no reported run starts until all three are frozen.
  Reported   the chosen setting at seeds 42, 123, 456, 789 and 2026. Its seed-42 run IS its
             selection run: that run's test value is computed only after all three choices
             are frozen, by scoring its saved checkpoint (grid-test), and every comparison
             is also reported without seed 42.
  Stopping   every run: validation every epoch through epoch 20 and every 5th epoch
             after; patience 20 epochs since the best evaluation, but never before epoch
             45; cap 200.
  Shared     dim 128, 3 layers, batch 32,768, Adam 1e-3, L2 1e-5, one sampled negative.
             Initialisation is Xavier-uniform, as all three authors use.

    python3 scripts/contrastive_grid.py plan               every run, numbered, and its state
    python3 scripts/contrastive_grid.py select --only 3    one selection run (#1-#99)
    python3 scripts/contrastive_grid.py pick               advance each search; freeze each choice
    python3 scripts/contrastive_grid.py seeds --only 101   one reported run (#101-#112)
    python3 scripts/contrastive_grid.py grid-test          after the freeze: every selection run on test
    python3 scripts/contrastive_grid.py collect            the 5 reported seeds per model, with checksums

Runs already finished are skipped, an interrupted run resumes from its training_state.pt,
and a run that is already going is refused, so any command can simply be repeated.
Anything after `--` is passed to every run_experiment.py call (e.g. `-- --data-dir ...`).

The protocol is for the ML-20M split only. Before it launches a run, and before it trusts
a finished one, it checks the split and the training settings. A trial on another split
or a shorter budget needs --smoke, which also writes to separate directories.

The frozen choices this search produced, with their SHA-256, are
benchmark/contrastive_settings.json; run_experiment.py applies them to M1b-M1d.

Exit codes: 0 done; 1 a run failed or the self-check failed; 2 a precondition is missing.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import REPO, code  # noqa: E402

BENCH = code("benchmark")
ENCODER = "bge-large-en-v1.5"       # the path component run_experiment.py uses for ID-only configs
SELECT_SEED = 42
REPORT_SEEDS = (42, 123, 456, 789, 2026)       # the five seeds every configuration reports
NEW_SEEDS = tuple(s for s in REPORT_SEEDS if s != SELECT_SEED)   # trained after the freeze
FIRST_REPORTED = 101
MIN_EPOCHS, DENSE_EVAL_UNTIL, EVAL_EVERY = 45, 20, 5
RUN_FLAGS = ["--min-epochs", str(MIN_EPOCHS), "--dense-eval-until", str(DENSE_EVAL_UNTIL)]

# Sources, checked against the papers and the official code:
#  SimGCL (SIGIR'22): lambda explored {0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5}; best 0.2, 0.5 and 2
#    on Douban-Book, Yelp2018 and Amazon-Book, and XSimGCL's Table V adds 0.1 and 0.05. eps is
#    best "near 0.1" on every dataset (0.1 in every Table V row); tau is fixed at 0.2.
#  XSimGCL (TKDE'23): lambda explored {0.01, 0.05, 0.1, 0.2, 0.5, 1}, best 0.05-0.2. The authors
#    tune lambda at l* = 1, then l* ("three attempts are enough" at 3 layers; best l* 1-4 by
#    dataset). eps: "0.1 is usually a good and safe choice", best 0.05-0.2 by dataset. tau 0.2.
#  LightGCL (ICLR'23), as released, which this harness follows: the README sets tau per dataset
#    (0.1, 0.2 by default, 0.5) and never changes lambda_1 (0.2) or q (5); the paper's older loss
#    explored tau 0.1-10.
# Each stage: (hyperparameter, searched values, ladder below the range, ladder above it).
SEARCH = {
    "m1b": {"fixed": {"cl_eps": 0.1, "cl_temp": 0.2}, "start": {},
            "stages": [("cl_weight", (0.05, 0.1, 0.2, 0.5), (0.02, 0.01), (1.0, 2.0))]},
    "m1c": {"fixed": {"cl_temp": 0.2}, "start": {"cl_eps": 0.1, "layer_cl": 1},
            "stages": [("cl_weight", (0.05, 0.1, 0.2), (0.02, 0.01), (0.5, 1.0)),
                       ("layer_cl", (1, 2, 3), (), ()),
                       ("cl_eps", (0.05, 0.1, 0.2), (), ())]},
    "m1d": {"fixed": {"cl_weight": 0.2}, "start": {},
            "stages": [("cl_temp", (0.1, 0.2, 0.5), (), (1.0,))]},
}
ORDER = ("cl_weight", "cl_eps", "cl_temp", "layer_cl")
FLAG = {"cl_weight": "--cl-weight", "cl_eps": "--cl-eps", "cl_temp": "--cl-temp",
        "layer_cl": "--layer-cl"}
SHORT = {"cl_weight": "lam", "cl_eps": "eps", "cl_temp": "tau", "layer_cl": "l"}
HPARAM_KEY = {"cl_eps": "noise_eps"}   # launcher name -> the name the model records
MODEL = {"m1b": "simgcl", "m1c": "xsimgcl", "m1d": "lightgcl"}
DEFAULT_MODELS = ",".join(SEARCH)
RULE = ("best validation NDCG@10 at seed 42; exact ties go to the smaller value; a winner at "
        "the end of a searched range extends along its ladder and the stage re-selects")

# The ML-20M split every other configuration in the paper trained on, as its
# stats.json describes it. A run on any other split is not a grid run.
ML20M = {"n_users": 127_371, "n_items": 9_906, "n_train": 11_499_778,
         "n_val": 49_668, "n_test": 67_466}


def protocol() -> dict:
    """The training settings every run must record: the harness's shared budget plus the
    stopping floor and evaluation cadence of this protocol."""
    sys.path.insert(0, str(BENCH))
    import config
    return {"lr": config.LEARNING_RATE, "weight_decay": config.WEIGHT_DECAY,
            "num_epochs": config.NUM_EPOCHS, "patience": config.PATIENCE,
            "batch_size": config.BATCH_SIZE, "min_epochs": MIN_EPOCHS,
            "dense_eval_until": DENSE_EVAL_UNTIL, "eval_every": EVAL_EVERY}


def canon(hp: dict) -> dict:
    return {k: hp[k] for k in ORDER if k in hp}


def tag(cfg: str, hp: dict) -> str:
    return cfg + "_" + "_".join(f"{SHORT[k]}{v:g}" for k, v in canon(hp).items()) + "_uniform"


def paths(out: Path, ckpt: Path, cfg: str, hp: dict, seed: int):
    t = tag(cfg, hp)
    res_dir, ckpt_dir = out / t, ckpt / t
    result = res_dir / cfg / ENCODER / f"seed-{seed}" / "results.json"
    log = out / "logs" / f"{t}_seed{seed}.log"
    return res_dir, ckpt_dir, result, log


def command(cfg, hp, seed, out, ckpt, device, extra, selection):
    res_dir, ckpt_dir, _, _ = paths(out, ckpt, cfg, hp, seed)
    cmd = [sys.executable, str(BENCH / "run_experiment.py"), "--config", cfg, "--seed", str(seed),
           "--device", device, "--init", "uniform",
           "--results-dir", str(res_dir), "--checkpoint-dir", str(ckpt_dir)] + RUN_FLAGS
    for k, v in canon(hp).items():
        cmd += [FLAG[k], str(v)]
    if selection:
        cmd.append("--no-test")
    return cmd + list(extra)


# ---------------------------------------------------------------- the search
def fresh_state() -> dict:
    """Runs #1-#10: the first stage of each model's search, numbered before any run."""
    runs, n = [], 1
    for cfg, spec in SEARCH.items():
        dim, values, _, _ = spec["stages"][0]
        for v in values:
            runs.append({"n": n, "cfg": cfg, "why": f"{dim} grid",
                         "hp": canon({**spec["fixed"], **spec["start"], dim: v})})
            n += 1
    return {"select": runs, "chosen": {}}


def load_state(out: Path) -> dict:
    f = out / "search.json"
    return json.loads(f.read_text()) if f.is_file() else fresh_state()


def save_state(out: Path, state: dict):
    out.mkdir(parents=True, exist_ok=True)
    (out / "search.json").write_text(json.dumps(state, indent=2) + "\n")


def reported_runs(state: dict) -> dict:
    """#101-#112: each model's chosen setting at the reported seeds other than 42 (None
    until frozen). Its seed-42 reported run is the selection run itself."""
    runs, n = {}, FIRST_REPORTED
    for cfg in SEARCH:
        chosen = (state["chosen"].get(cfg) or {}).get("hp")
        for s in NEW_SEEDS:
            runs[n] = (cfg, chosen, s)
            n += 1
    return runs


def advance(cfg: str, val_of):
    """Walk the pre-registered search for `cfg` as far as finished results allow.

    Deterministic: the same results always give the same decisions. Returns
    ("chosen", hp, trace) or ("waiting", [hp, ...], trace), where the list holds the
    points whose results the next decision needs.
    """
    spec = SEARCH[cfg]
    cur = {**spec["fixed"], **spec["start"]}
    trace = []
    for dim, values, below, above in spec["stages"]:
        tried = list(values)
        while True:
            grid = sorted(tried)
            pts = [canon({**cur, dim: v}) for v in grid]
            missing = [p for p in pts if val_of(p) is None]
            if missing:
                return "waiting", missing, trace
            scores = [val_of(p) for p in pts]
            best = max(range(len(grid)), key=lambda i: (scores[i], -grid[i]))
            v = grid[best]
            ladder = below if best == 0 else above if best == len(grid) - 1 else ()
            nxt = next((x for x in ladder if x not in tried), None)
            if nxt is not None:
                trace.append(f"{dim}: best {v:g} is at the end of {grid}; extending to {nxt:g}")
                tried.append(nxt)
                continue
            edge = "" if 0 < best < len(grid) - 1 else (
                " -- at the end of its ladder, so not bracketed" if ladder else
                " -- at the end of the searched range, which the protocol does not extend")
            trace.append(f"{dim}: chose {v:g} from {grid}"
                         f" (val NDCG@10 {scores[best]:.5f}){edge}")
            cur[dim] = v
            break
    return "chosen", canon(cur), trace


def find_run(state: dict, cfg: str, hp: dict):
    return next((r for r in state["select"] if r["cfg"] == cfg and r["hp"] == canon(hp)), None)


# ---------------------------------------------------------------- checks
def problems_with(result: Path, cfg: str, hp: dict, selection: bool) -> list[str]:
    """Why a finished result cannot stand as this run of the protocol; [] if it can."""
    try:
        r = json.loads(result.read_text())
    except (OSError, ValueError) as e:
        return [f"unreadable ({e})"]
    out = []
    got = r.get("config") or {}
    for k, want in protocol().items():
        if k not in got:
            out.append(f"{k} not recorded")
        elif not math.isclose(float(got[k]), float(want), rel_tol=1e-9):
            out.append(f"{k}={got[k]} (protocol {want})")
    data = r.get("data") or {}
    for k, want in ML20M.items():
        if data.get(k) != want:
            out.append(f"{k}={data.get(k)} (ML-20M {want})")
    mh = r.get("model_hparams") or {}
    if mh.get("model") != MODEL[cfg]:
        out.append(f"model={mh.get('model')} (expected {MODEL[cfg]})")
    if mh.get("init") != "uniform":
        out.append(f"init={mh.get('init')} (expected uniform)")
    for k, v in canon(hp).items():
        name = HPARAM_KEY.get(k, k)
        if name not in mh or not math.isclose(float(mh[name]), float(v), rel_tol=1e-9):
            out.append(f"{name}={mh.get(name)} (protocol {v})")
    if selection and r.get("test_metrics") is not None:
        out.append("a selection run carries test metrics (it must be run with --no-test)")
    if not selection and not (r.get("test_metrics") or {}).get("NDCG@10"):
        out.append("a reported run has no test metrics")
    return out


def split_problems(data_dir: Path) -> list[str]:
    """Why data_dir is not the ML-20M split; [] if it is."""
    stats = data_dir / "stats.json"
    if not stats.is_file():
        return [f"no stats.json at {data_dir} (fetch or build the ML-20M splits first)"]
    s = json.loads(stats.read_text())
    return [f"{k}={s.get(k)} (ML-20M {want})" for k, want in ML20M.items() if s.get(k) != want]


def data_dir_of(extra) -> Path:
    if "--data-dir" in extra[:-1]:
        return Path(extra[extra.index("--data-dir") + 1])
    sys.path.insert(0, str(BENCH))
    import config
    return Path(config.DATA_DIR)


def self_check() -> list[str]:
    """The contrastive objective is active: each model's contrastive term moves the gradient.

    A model whose contrastive term does not move the gradient trains as LightGCN under
    another name; this refuses to launch a single run on such a checkout. Tiny CPU
    graph, about a second.
    """
    import warnings
    import torch
    import torch.nn.functional as F
    warnings.filterwarnings("ignore", message="Sparse invariant checks")
    sys.path.insert(0, str(BENCH))
    from models.simgcl import SimGCL
    from models.xsimgcl import XSimGCL
    from models.lightgcl import LightGCL
    nu, ni, d, layers = 40, 30, 8, 3
    torch.manual_seed(0)
    r, c = torch.randint(0, nu, (300,)), torch.randint(0, ni, (300,))
    idx = torch.cat([torch.stack([r, c + nu]), torch.stack([c + nu, r])], 1)
    a = torch.sparse_coo_tensor(idx, torch.ones(idx.shape[1]), (nu + ni, nu + ni)).coalesce()
    dinv = torch.sparse.sum(a, 1).to_dense().clamp(min=1).pow(-0.5)
    i = a.indices()
    adj = torch.sparse_coo_tensor(i, dinv[i[0]] * a.values() * dinv[i[1]], a.shape).coalesce()
    problems = []
    for cls in (SimGCL, XSimGCL, LightGCL):
        grads = []
        for w in (0.0, 0.5):
            torch.manual_seed(1)
            m = cls(nu, ni, d, layers, cl_weight=w)
            m.set_adj(adj)
            u, p, n = (torch.randint(0, x, (32,)) for x in (nu, ni, ni))
            torch.manual_seed(2)
            o = m(u, p, n)
            if len(o) != 4:
                problems.append(f"{cls.__name__} returns {len(o)} outputs; the models return 4")
                break
            loss = -F.logsigmoid(o[0] - o[1]).mean() + 1e-5 * o[2] + o[3]
            grads.append(torch.autograd.grad(loss, m.user_emb.weight)[0])
        if len(grads) == 2 and (grads[1] - grads[0]).norm().item() < 1e-3:
            problems.append(f"{cls.__name__}'s contrastive term does not move the gradient")
    return problems


def checkout_state() -> str:
    """The commit these runs train from, and whether tracked files differ from it."""
    try:
        git = ["git", "-C", str(REPO)]
        sha = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(git + ["status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + (" with uncommitted changes" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown (not a git checkout)"


def environment(gpu=None) -> str:
    """Interpreter, PyTorch, CUDA and the device a run will use."""
    import platform
    import torch
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_name(int(gpu) if gpu not in (None, "") else 0)
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev = "MPS"
    else:
        dev = "CPU"
    return (f"python {platform.python_version()} | torch {torch.__version__} | "
            f"cuda {torch.version.cuda} | {dev}")


# ---------------------------------------------------------------- locks and the queue
def lock_path(log: Path) -> Path:
    return log.with_suffix(".lock")


def lock_holder(lock: Path):
    """Who is running this run now, or None. The lock names the host and the PIDs of
    the launcher and of its training process; the run is live if either is."""
    try:
        host, *pids = lock.read_text().split()
        pids = [int(x) for x in pids]
    except (OSError, ValueError):
        return None
    if not pids:
        return None
    if host != socket.gethostname():
        return f"PID {pids[-1]} on host {host} (cannot be checked from here)"
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            pass
        return f"PID {pid} on {host}"
    return None


def acquire(lock: Path):
    """Take a run's lock. Returns None when taken, else who holds it. A lock whose
    processes are all gone (a crash, a reboot) is stale and is taken over."""
    lock.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            holder = lock_holder(lock)
            if holder:
                return holder
            lock.unlink(missing_ok=True)
            continue
        with os.fdopen(fd, "w") as fh:
            fh.write(f"{socket.gethostname()} {os.getpid()}\n")
        return None
    return "another launcher, which took it first"


def run_queue(jobs, gpus, out, checkout):
    """Run jobs, one per GPU slot at a time. Returns (failed, busy) labels."""
    (out / "logs").mkdir(parents=True, exist_ok=True)
    slots = list(gpus) if gpus else [None]
    free = list(slots)
    lock = threading.Lock()
    failed, busy = [], []
    envs = {}

    def work(job):
        cmd, log, label = job
        run_lock = lock_path(log)
        holder = acquire(run_lock)
        if holder:
            with lock:
                busy.append(label)
            print(f"  BUSY   {label}: already running as {holder}", flush=True)
            return
        with lock:
            gpu = free.pop(0)
            if gpu not in envs:
                try:
                    envs[gpu] = environment(gpu)
                except Exception as e:  # the header is provenance; never block a run on it
                    envs[gpu] = f"unknown ({type(e).__name__})"
        rc = None
        try:
            env = dict(os.environ)
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print(f"  start  {label}" + (f" on GPU {gpu}" if gpu is not None else ""), flush=True)
            with open(log, "a") as fh:
                fh.write(f"# checkout: {checkout}\n# env: {envs[gpu]}\n$ " + " ".join(cmd) + "\n")
                fh.flush()
                proc = subprocess.Popen(cmd, cwd=BENCH, env=env, stdout=fh, stderr=subprocess.STDOUT)
                run_lock.write_text(f"{socket.gethostname()} {os.getpid()} {proc.pid}\n")
                rc = proc.wait()
        finally:
            run_lock.unlink(missing_ok=True)
            with lock:
                free.append(gpu)
                if rc != 0:
                    failed.append(label)
        print(f"  {'done  ' if rc == 0 else 'FAILED'} {label} (exit {rc}; log {log})", flush=True)

    with ThreadPoolExecutor(max_workers=len(slots)) as ex:
        list(ex.map(work, jobs))
    return failed, busy


# ---------------------------------------------------------------- small helpers
def val_ndcg(result: Path):
    if not result.is_file():
        return None
    r = json.loads(result.read_text())
    return (r.get("best_val_metrics") or {}).get("NDCG@10")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def header_lines(log: Path, prefix: str) -> list[str]:
    if not log.is_file():
        return []
    seen = []
    for line in log.read_text(errors="ignore").splitlines():
        if line.startswith(prefix) and line[len(prefix):] not in seen:
            seen.append(line[len(prefix):])
    return seen


def checkouts_in(log: Path) -> list[str]:
    return header_lines(log, "# checkout: ")


def refuse(title: str, rows: list[str], hint: str) -> int:
    print(f"REFUSED: {title}", file=sys.stderr)
    for row in rows:
        print("  - " + row, file=sys.stderr)
    print(hint, file=sys.stderr)
    return 2


def chosen_digest(state: dict) -> str:
    return hashlib.sha256(json.dumps(state["chosen"], sort_keys=True).encode()).hexdigest()


NOT_A_GRID_RUN = ("They look like a trial run. Move them out of --out, or use a fresh --out and\n"
                  "--ckpt; a trial belongs under --smoke, which writes elsewhere by default.")


def settings(hp: dict) -> str:
    return ", ".join(f"{SHORT[k]} {v:g}" for k, v in canon(hp).items())


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    if sys.version_info < (3, 10):
        print(f"Python 3.10 or newer is required (the project's requires-python); this is "
              f"{sys.version.split()[0]}", file=sys.stderr)
        return 2
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("phase", choices=["plan", "status", "select", "pick", "seeds",
                                      "grid-test", "collect"])
    ap.add_argument("--out", type=Path, default=None,
                    help="result files, logs and the search record "
                         "(default results_contrastive/, or results_contrastive_smoke/ with --smoke)")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="checkpoints, large, keep them off git "
                         "(default checkpoints_contrastive/, or checkpoints_contrastive_smoke/)")
    ap.add_argument("--final", type=Path, default=None,
                    help="where collect writes (default <out>/final)")
    ap.add_argument("--gpus", default="",
                    help="comma-separated GPU ids; one run per id at a time "
                         "(repeat an id, e.g. 0,0, to run two at once on one GPU)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models", default=DEFAULT_MODELS, help="subset, e.g. m1b")
    ap.add_argument("--only", default="",
                    help="only these runs, by the number `plan` prints, e.g. 3 or 1,2 "
                         "(selection runs are #1-#99, reported runs #101-#112)")
    ap.add_argument("--smoke", action="store_true",
                    help="a trial: lift the ML-20M and training-settings checks")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    # Everything after `--` goes to every run. Split it off before argparse, which
    # would otherwise let a catch-all positional swallow options such as --gpus.
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    a = ap.parse_args(argv)
    suffix = "_smoke" if a.smoke else ""
    a.out = a.out or REPO / f"results_contrastive{suffix}"
    a.ckpt = a.ckpt or REPO / f"checkpoints_contrastive{suffix}"
    # Runs execute from the benchmark directory, so a relative --data-dir or
    # --embedding-dir would resolve against the wrong folder. Anchor such paths to
    # the directory the launcher was started from (either spelling, --flag X or
    # --flag=X).
    extra = [x for arg in extra for x in
             (arg.split("=", 1) if arg.startswith(("--data-dir=", "--embedding-dir=")) else [arg])]
    for i, arg in enumerate(extra[:-1]):
        if arg in ("--data-dir", "--embedding-dir") and not os.path.isabs(extra[i + 1]):
            extra[i + 1] = os.path.abspath(extra[i + 1])
    models = [m for m in a.models.split(",") if m]
    unknown = [m for m in models if m not in SEARCH]
    if unknown:
        print(f"unknown model(s) {unknown}; choose from {sorted(SEARCH)}", file=sys.stderr)
        return 2
    gpus = [g for g in a.gpus.split(",") if g != ""]
    state = load_state(a.out)
    frozen = all(cfg in state["chosen"] for cfg in SEARCH)

    def check(result, cfg, hp, selection):
        return [] if a.smoke else problems_with(result, cfg, hp, selection)

    only = []
    if a.only:
        if a.phase not in ("select", "seeds"):
            print("--only applies to select and seeds", file=sys.stderr)
            return 2
        if a.models != DEFAULT_MODELS:
            print("use --only or --models, not both", file=sys.stderr)
            return 2
        try:
            only = sorted({int(x) for x in a.only.split(",") if x.strip()})
        except ValueError:
            print(f"--only takes run numbers, e.g. 3 or 1,2; got {a.only!r}", file=sys.stderr)
            return 2
        known = ({r["n"] for r in state["select"]} if a.phase == "select"
                 else set(reported_runs(state)))
        if not only or any(n not in known for n in only):
            kind = ("registered selection runs (`pick` registers later stages and edge "
                    "extensions)" if a.phase == "select" else "reported runs #101-#112")
            print(f"--only {a.only}: not all {kind}; `plan` lists them", file=sys.stderr)
            return 2


    # -------------------------------------------------------------- plan / status
    if a.phase in ("plan", "status"):
        def state_of(result, log, cfg, hp, selection):
            v = val_ndcg(result)
            if v is None:
                holder = lock_holder(lock_path(log))
                if holder:
                    return f"running ({holder})"
                if log.is_file():
                    return "started, no result: failed or interrupted; its log says which, and repeating it resumes"
                return "not run"
            p = check(result, cfg, hp, selection)
            return f"val NDCG@10 {v:.5f}" + (f"  NOT A PROTOCOL RUN: {'; '.join(p)}" if p else "")

        print(f"  selection runs (seed {SELECT_SEED}; no test value until all three are frozen):")
        for r in state["select"]:
            if r["cfg"] not in models:
                continue
            _, _, result, log = paths(a.out, a.ckpt, r["cfg"], r["hp"], SELECT_SEED)
            print(f"  #{r['n']:<3} {r['cfg']:<4} {settings(r['hp']):<34} "
                  f"{state_of(result, log, r['cfg'], r['hp'], True)}")
        print("  searches:")
        for cfg in models:
            if cfg in state["chosen"]:
                print(f"    {cfg}: FROZEN at {settings(state['chosen'][cfg]['hp'])}")
                continue

            def peek(hp, cfg=cfg):
                run = find_run(state, cfg, hp)
                return None if run is None else val_ndcg(paths(a.out, a.ckpt, cfg, hp, SELECT_SEED)[2])
            status, payload, _ = advance(cfg, peek)
            if status == "chosen":
                print(f"    {cfg}: ready to freeze at {settings(payload)} -- run `pick`")
            else:
                ids = [find_run(state, cfg, hp) for hp in payload]
                if all(ids):
                    print(f"    {cfg}: waiting for run(s) " + ", ".join(f"#{r['n']}" for r in ids))
                else:
                    print(f"    {cfg}: `pick` will register the next runs")
        print(f"  reported runs (the frozen settings; seed {SELECT_SEED} of each is its selection run):")
        for cfg in models:
            if cfg in state["chosen"]:
                r = find_run(state, cfg, state["chosen"][cfg]["hp"])
                print(f"  seed {SELECT_SEED} of {cfg} = selection run #{r['n']} "
                      f"(scored on test by grid-test, after the freeze)")
        for n, (cfg, hp, s) in reported_runs(state).items():
            if cfg not in models:
                continue
            if hp is None:
                print(f"  #{n:<3} {cfg:<4} {'(after all three are frozen)':<34} seed {s}")
                continue
            _, _, result, log = paths(a.out, a.ckpt, cfg, hp, s)
            print(f"  #{n:<3} {cfg:<4} {settings(hp):<34} seed {s:<5} "
                  f"{state_of(result, log, cfg, hp, False)}")
        print("  One run: `select --only N` or `seeds --only N`, followed by "
              "`-- --data-dir <the ML-20M split>`.")
        return 0

    # -------------------------------------------------------------- pick
    if a.phase == "pick":
        bad = []

        def val_of(cfg, hp):
            run = find_run(state, cfg, hp)
            if run is None:
                return None
            result = paths(a.out, a.ckpt, cfg, hp, SELECT_SEED)[2]
            if not result.is_file():
                return None
            if problems := check(result, cfg, hp, True):
                bad.append(f"#{run['n']} {result}: {'; '.join(problems)}")
                return None
            return val_ndcg(result)

        report = []
        for cfg in models:
            if cfg in state["chosen"]:
                report.append(f"  {cfg}: frozen at {settings(state['chosen'][cfg]['hp'])} "
                              f"(a frozen choice is never revisited)")
                continue
            status, payload, trace = advance(cfg, lambda hp, cfg=cfg: val_of(cfg, hp))
            if bad:
                return refuse("these selection results are not protocol runs:", bad, NOT_A_GRID_RUN)
            if status == "chosen":
                runs = [r["n"] for r in state["select"] if r["cfg"] == cfg]
                state["chosen"][cfg] = {
                    "hp": payload, "trace": trace, "selection_runs": runs,
                    "frozen_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
                        timespec="seconds")}
                report.append(f"  {cfg}: FROZEN at {settings(payload)}")
                report += [f"      {line}" for line in trace]
                continue
            new = []
            for hp in payload:
                if find_run(state, cfg, hp) is None:
                    n = max(r["n"] for r in state["select"]) + 1
                    why = trace[-1] if trace and "extending" in trace[-1] else "next stage"
                    state["select"].append({"n": n, "cfg": cfg, "hp": hp, "why": why})
                    new.append(n)
            waiting = [find_run(state, cfg, hp)["n"] for hp in payload]
            report.append(f"  {cfg}: waiting for run(s) " + ", ".join(f"#{n}" for n in waiting)
                          + (f" (newly registered: {', '.join(f'#{n}' for n in new)})" if new else ""))
            report += [f"      {line}" for line in trace]
            for n in waiting:
                r = next(x for x in state["select"] if x["n"] == n)
                report.append(f"      #{n}: {settings(r['hp'])} -- start it with "
                              f"`select --only {n} -- --data-dir <the ML-20M split>`")
        if any(n >= FIRST_REPORTED for n in (r["n"] for r in state["select"])):
            print("more selection runs than the numbering allows", file=sys.stderr)
            return 2
        state["rule"] = RULE
        if all(cfg in state["chosen"] for cfg in SEARCH):
            state["chosen_sha256"] = chosen_digest(state)
        save_state(a.out, state)
        print("\n".join(report))
        if "chosen_sha256" in state:
            print(f"  All three selections are frozen. Record: {a.out / 'search.json'}")
            print(f"  SHA-256 of the frozen choices: {state['chosen_sha256']}")
            print("  Record that hash before starting any reported run (runs #101-#112).")
        return 0

    # -------------------------------------------------------------- grid-test
    if a.phase == "grid-test":
        if not frozen:
            return refuse("selections are still open:", [c for c in SEARCH if c not in state["chosen"]],
                          "No selection run is evaluated on test until all three choices are frozen.")
        data_dir = data_dir_of(extra)
        if not a.smoke and (p := split_problems(data_dir)):
            return refuse(f"{data_dir} is not the ML-20M split:", p, "")
        import warnings
        import torch
        warnings.filterwarnings("ignore", message="Sparse invariant checks")
        sys.path.insert(0, str(BENCH))
        from data.dataset import InteractionData
        from evaluate import evaluate_model
        from run_experiment import build_model
        data = InteractionData(data_dir=data_dir)
        norm_adj = data.get_norm_adj().to(a.device)
        rows = []
        for r in state["select"]:
            if r["cfg"] not in models:
                continue
            _, ckpt_dir, result, _ = paths(a.out, a.ckpt, r["cfg"], r["hp"], SELECT_SEED)
            weights = ckpt_dir / r["cfg"] / ENCODER / f"seed-{SELECT_SEED}" / "best_model.pt"
            if not weights.is_file():
                continue
            name = MODEL[r["cfg"]]
            if name == "lightgcl":
                m = build_model(name, data.n_users, data.n_items, 0, None)
                m.norm_adj = norm_adj
            else:
                m = build_model(name, data.n_users, data.n_items, 0, norm_adj)
            m.load_state_dict(torch.load(weights, map_location=a.device, weights_only=True))
            m.to(a.device)
            tm = evaluate_model(m, data, split="test", device=a.device)
            chosen = state["chosen"][r["cfg"]]["hp"] == r["hp"]
            rows.append({"n": r["n"], "cfg": r["cfg"], "hp": r["hp"], "chosen": chosen,
                         "val_ndcg10": val_ndcg(result),
                         "test_metrics": {k: float(v) for k, v in tm.items()},
                         "best_model_sha256": sha256(weights)})
            print(f"  #{r['n']:<3} {r['cfg']:<4} {settings(r['hp']):<34} val {rows[-1]['val_ndcg10']:.5f}"
                  f"  test {tm['NDCG@10']:.5f}{'   <- chosen' if chosen else ''}")
        (a.out / "grid_test.json").write_text(json.dumps(
            {"chosen_sha256": state.get("chosen_sha256"), "runs": rows}, indent=2) + "\n")
        print(f"  written {a.out / 'grid_test.json'}")
        return 0

    # -------------------------------------------------------------- collect
    if a.phase == "collect":
        if not frozen:
            return refuse("selections are still open:", [c for c in SEARCH if c not in state["chosen"]],
                          "Collect after all three choices are frozen and their runs are done.")
        final = a.final or a.out / "final"
        manifest_path = final / "MANIFEST.json"
        # collect adds models to a folder it wrote before (its MANIFEST says so) and
        # never touches one it did not write.
        if final.exists() and any(final.iterdir()) and not manifest_path.is_file():
            print(f"{final} holds files that collect did not write; remove them or pass "
                  f"another --final", file=sys.stderr)
            return 2
        prev = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        if prev and bool(prev.get("smoke")) != a.smoke:
            print(f"{final} holds {'trial' if prev.get('smoke') else 'real'} runs; trial and real "
                  f"runs never share a folder", file=sys.stderr)
            return 2
        grid_file = a.out / "grid_test.json"
        if not grid_file.is_file():
            print("run grid-test first: it scores each chosen setting's seed-42 selection run on "
                  "test, and that is its seed-42 reported value", file=sys.stderr)
            return 2
        graded = {row["n"]: row for row in json.loads(grid_file.read_text())["runs"]}
        missing, bad, runs = [], [], []
        for cfg in models:                      # seed 42: the frozen setting's selection run
            hp = state["chosen"][cfg]["hp"]
            run = find_run(state, cfg, hp)
            _, ckpt_dir, result, log = paths(a.out, a.ckpt, cfg, hp, SELECT_SEED)
            weights = ckpt_dir / cfg / ENCODER / f"seed-{SELECT_SEED}" / "best_model.pt"
            row = graded.get(run["n"])
            if not result.is_file() or not weights.is_file() or row is None:
                missing.append(f"#{run['n']} {cfg} seed {SELECT_SEED} (its selection run, "
                               f"scored by grid-test)")
                continue
            if problems := check(result, cfg, hp, True):
                bad.append(f"#{run['n']} {result}: {'; '.join(problems)}")
            if row["best_model_sha256"] != sha256(weights):
                bad.append(f"#{run['n']}: the checkpoint changed after grid-test scored it")
            runs.append((run["n"], cfg, hp, SELECT_SEED, result, weights, log, row["test_metrics"]))
        for n, (cfg, hp, s) in reported_runs(state).items():
            if cfg not in models:
                continue
            _, ckpt_dir, result, log = paths(a.out, a.ckpt, cfg, hp, s)
            weights = ckpt_dir / cfg / ENCODER / f"seed-{s}" / "best_model.pt"
            if not result.is_file() or not weights.is_file():
                missing.append(f"#{n} {cfg} seed {s}")
                continue
            if problems := check(result, cfg, hp, False):
                bad.append(f"#{n} {result}: {'; '.join(problems)}")
            runs.append((n, cfg, hp, s, result, weights, log, None))
        if missing:
            print(f"cannot collect: {len(missing)} reported run(s) are not finished: "
                  + ", ".join(missing), file=sys.stderr)
            return 2
        if bad:
            return refuse("these results are not protocol runs:", bad, NOT_A_GRID_RUN)
        manifest = {
            "what": ("M1b SimGCL, M1c XSimGCL and M1d LightGCL, trained with their authors' "
                     "objectives on the ML-20M split: each method's own settings selected on "
                     "validation NDCG@10 at seed 42, then trained at seeds "
                     + ", ".join(map(str, NEW_SEEDS)) + ". The seed-42 reported run is the "
                     "selection run, scored on test only after all three choices were frozen."),
            "sensitivity": ("the seed-42 value of each setting was selected for its validation "
                            "score; every comparison is also reported without seed 42 (n = 4)"),
            "rule": RULE, "protocol": protocol(), "data": ML20M, "smoke": a.smoke,
            "chosen_sha256": state.get("chosen_sha256"),
            "configs": {k: v for k, v in (prev.get("configs") or {}).items() if k not in models}}
        # A model collected again replaces its earlier copy entirely.
        for cfg in models:
            old = (prev.get("configs") or {}).get(cfg)
            if old:
                shutil.rmtree(final / "results" / cfg, ignore_errors=True)
                shutil.rmtree(final / "checkpoints" / cfg, ignore_errors=True)
                for s in old["seeds"]:
                    (final / "logs" / f"{old['tag']}_seed{s}.log").unlink(missing_ok=True)
        for n, cfg, hp, s, result, weights, log, graded_test in runs:
            leaf = Path(cfg) / ENCODER / f"seed-{s}"
            (final / "results" / leaf).mkdir(parents=True, exist_ok=True)
            (final / "checkpoints" / leaf).mkdir(parents=True, exist_ok=True)
            r = json.loads(result.read_text())
            if graded_test is not None:
                r["test_metrics"] = graded_test
                r["_provenance"] = ("the selection run of the frozen setting; its test metrics come "
                                    "from grid-test's eval-only re-scoring of this checkpoint, made "
                                    "after all three choices were frozen")
            (final / "results" / leaf / "results.json").write_text(json.dumps(r, indent=2) + "\n")
            shutil.copy2(weights, final / "checkpoints" / leaf / "best_model.pt")
            meta = weights.with_name("best_model.meta.json")
            if meta.is_file():
                shutil.copy2(meta, final / "checkpoints" / leaf / meta.name)
            if log.is_file():
                (final / "logs").mkdir(parents=True, exist_ok=True)
                shutil.copy2(log, final / "logs" / log.name)
            entry = manifest["configs"].setdefault(cfg, {
                "tag": tag(cfg, hp), "hparams": hp, "search": state["chosen"][cfg],
                "selection_runs": [
                    {"n": x["n"], "hp": x["hp"], "val_ndcg10":
                     val_ndcg(paths(a.out, a.ckpt, cfg, x["hp"], SELECT_SEED)[2])}
                    for x in state["select"] if x["cfg"] == cfg],
                "seeds": {}})
            entry["seeds"][str(s)] = {
                "run": n,
                "selection_run": graded_test is not None,
                "val_ndcg10": (r.get("best_val_metrics") or {}).get("NDCG@10"),
                "test_ndcg10": (r.get("test_metrics") or {}).get("NDCG@10"),
                "best_epoch": r.get("best_epoch"),
                "total_epochs_trained": r.get("total_epochs_trained"),
                "stopped_early": r.get("stopped_early"),
                "best_model_sha256": sha256(weights),
                "trained_from": checkouts_in(log),
                "environment": header_lines(log, "# env: "),
            }
        shutil.copy2(grid_file, final / "grid_test.json")
        shutil.copy2(a.out / "search.json", final / "search.json")
        (final / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for cfg, entry in manifest["configs"].items():
            if cfg not in models:
                continue
            tests = [v["test_ndcg10"] for v in entry["seeds"].values() if v["test_ndcg10"] is not None]
            print(f"  {cfg} {entry['tag']}: test NDCG@10 mean {sum(tests) / len(tests):.4f} over "
                  f"{len(tests)} seeds ({', '.join(f'{t:.4f}' for t in tests)})")
        mixed = sorted({c for e in manifest["configs"].values() for v in e["seeds"].values()
                        for c in v["trained_from"]})
        if len(mixed) > 1:
            print("  NOTE: these runs trained from more than one checkout: " + "; ".join(mixed))
        held = sorted(manifest["configs"], key=list(SEARCH).index)
        print(f"  written {final}: {len(runs)} run(s) now; it holds {', '.join(held)}"
              + ("" if len(held) == len(SEARCH) else f" of {', '.join(SEARCH)}"))
        return 0

    # -------------------------------------------------------------- select / seeds
    selection = a.phase == "select"
    if selection:
        plan = [(r["n"], r["cfg"], r["hp"], SELECT_SEED) for r in state["select"]]
    else:
        if not frozen:
            return refuse("selections are still open:", [c for c in SEARCH if c not in state["chosen"]],
                          "Reported runs start only when all three choices are frozen (`pick`), so "
                          "no test value of a trained model exists while a selection is open.")
        plan = [(n, cfg, hp, s) for n, (cfg, hp, s) in reported_runs(state).items()]
    plan = [p for p in plan if p[1] in models and (not only or p[0] in only)]
    jobs, bad = [], []
    for n, cfg, hp, seed in plan:
        _, _, result, log = paths(a.out, a.ckpt, cfg, hp, seed)
        label = f"#{n} {tag(cfg, hp)} seed {seed}"
        if result.is_file():
            if problems := check(result, cfg, hp, selection):
                bad.append(f"{result}: {'; '.join(problems)}")
            else:
                print(f"  skip   {label} (finished)")
            continue
        jobs.append((command(cfg, hp, seed, a.out, a.ckpt, a.device, extra, selection), log, label))
    if bad:
        return refuse("finished results in --out are not protocol runs, so they cannot be "
                      "skipped as done:", bad, NOT_A_GRID_RUN)
    if jobs and not a.smoke:
        data_dir = data_dir_of(extra)
        if p := split_problems(data_dir):
            return refuse(f"{data_dir} is not the ML-20M split:", p,
                          "The protocol is for ML-20M only; a trial on another split needs --smoke.")
    running = [f"{label}: {h}" for _, log, label in jobs if (h := lock_holder(lock_path(log)))]
    if running:
        return refuse("these runs are already going:", running,
                      "Wait for them, or leave them out; `status` shows every run. A lock left by a\n"
                      "crash clears itself once its processes are gone.")
    if a.dry_run:
        for cmd, log, label in jobs:
            print(f"  {label}\n    {' '.join(cmd)}\n    log: {log}")
        return 0
    if not jobs:
        print("  nothing to run")
        return 0
    problems = self_check()
    if problems:
        print("REFUSED: this checkout's contrastive objective is not active:", file=sys.stderr)
        for p in problems:
            print("  - " + p, file=sys.stderr)
        return 1
    save_state(a.out, state)      # the numbering a run was launched under is on disk
    print(f"  self-check passed; {len(jobs)} run(s) on {'GPUs ' + ','.join(gpus) if gpus else a.device}")
    failed, busy = run_queue(jobs, gpus, a.out, checkout_state())
    if failed:
        print(f"{len(failed)} run(s) failed: " + "; ".join(failed), file=sys.stderr)
        return 1
    if busy:
        print(f"{len(busy)} run(s) were started elsewhere meanwhile and left alone: "
              + "; ".join(busy), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
