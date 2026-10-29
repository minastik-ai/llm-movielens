"""The contrastive baselines train the objective their names claim.

For a year the three contrastive models returned their contrastive loss inside the
regulariser slot, which the trainer multiplies by the weight decay (1e-5). The term
was computed every step and trained at a weight of about 2e-6, so the models were
LightGCN in all but name, and every result-level check passed because the numbers
were faithfully those of the runs. These tests check the objective itself:

  - SimGCL equals a transcription of the authors' implementation (SELFRec,
    model/graph/SimGCL.py and util/loss_torch.py) on a small random graph, with the
    same random draws: the clean pass, the perturbed pass, the noise, the InfoNCE
    value and gradient, and the whole contrastive term;
  - the trainer adds the returned contrastive term at full weight;
  - the contrastive term changes the embedding gradient (the delta is not zero);
  - XSimGCL equals a transcription of the authors' SELFRec XSimGCL (one perturbed
    pass for both losses, contrast with layer l*), with the same random draws;
  - LightGCL equals a transcription of the authors' HKUDS/LightGCL model.py (SVD view
    from the main view's previous layer, sums over layers, unnormalised InfoNCE over
    every node), given the same SVD factors; its chunked log-sum-exp equals the
    unchunked one, value and gradient.
"""
import json

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _layout import load_module  # noqa: E402


def import_module(name):
    """`models.simgcl` -> benchmark/models/simgcl.py, in either layout."""
    return load_module("benchmark", name.replace(".", "/") + ".py")

NU, NI, D, L = 60, 40, 16, 3
EPS, LAM, TAU = 0.1, 0.5, 0.2


def _graph():
    torch.manual_seed(0)
    r = torch.randint(0, NU, (500,))
    c = torch.randint(0, NI, (500,))
    idx = torch.cat([torch.stack([r, c + NU]), torch.stack([c + NU, r])], 1)
    a = torch.sparse_coo_tensor(idx, torch.ones(idx.shape[1]), (NU + NI, NU + NI)).coalesce()
    dinv = torch.sparse.sum(a, 1).to_dense().clamp(min=1).pow(-0.5)
    i = a.indices()
    return torch.sparse_coo_tensor(i, dinv[i[0]] * a.values() * dinv[i[1]], a.shape).coalesce()


def _ref_forward(adj, u, i, perturbed):
    """SELFRec SimGCL_Encoder.forward, with .cuda() removed."""
    ego = torch.cat([u, i], 0)
    layers = []
    for _ in range(L):
        ego = torch.sparse.mm(adj, ego)
        if perturbed:
            ego = ego + torch.sign(ego) * F.normalize(torch.rand_like(ego), dim=-1) * EPS
        layers.append(ego)
    out = torch.stack(layers, dim=1).mean(dim=1)
    return out[:NU], out[NU:]


def _ref_infonce(v1, v2, t):
    """SELFRec util/loss_torch.InfoNCE."""
    v1, v2 = F.normalize(v1, dim=1), F.normalize(v2, dim=1)
    return -torch.diag(F.log_softmax(v1 @ v2.T / t, dim=1)).mean()


@pytest.fixture
def simgcl():
    SimGCL = import_module("models.simgcl").SimGCL
    adj = _graph()
    m = SimGCL(NU, NI, D, L, noise_eps=EPS, cl_weight=LAM, cl_temp=TAU, cl_chunk=7)
    m.set_adj(adj)
    return m, adj


def test_simgcl_passes_equal_the_authors_encoder(simgcl):
    m, adj = simgcl
    u, i = m.user_emb.weight, m.item_emb.weight
    ours = torch.cat(m._propagate(False))
    ref = torch.cat(_ref_forward(adj, u, i, False))
    assert torch.allclose(ours, ref, atol=1e-6)
    torch.manual_seed(7)
    ours = torch.cat(m._propagate(True))
    torch.manual_seed(7)
    ref = torch.cat(_ref_forward(adj, u, i, True))
    assert torch.allclose(ours, ref, atol=1e-6)


def test_simgcl_noise_keeps_sign_and_has_norm_eps(simgcl):
    m, _ = simgcl
    torch.manual_seed(3)
    e = torch.randn(50, D)
    n = m._add_noise(e) - e
    assert torch.allclose(n.norm(dim=-1), torch.full((50,), EPS), atol=1e-6)
    assert bool(((n * e) >= 0).all())


def test_simgcl_infonce_equals_the_authors_value_and_gradient(simgcl):
    m, _ = simgcl
    a = torch.randn(33, D, requires_grad=True)
    b = torch.randn(33, D, requires_grad=True)
    ours = m._infonce_loss(a, b)
    g_ours = torch.autograd.grad(ours, (a, b))
    ref = _ref_infonce(a, b, TAU)
    g_ref = torch.autograd.grad(ref, (a, b))
    assert torch.allclose(ours, ref, atol=1e-6)
    assert torch.allclose(torch.cat(g_ours), torch.cat(g_ref), atol=1e-6)


def test_simgcl_contrastive_term_equals_the_authors(simgcl):
    m, adj = simgcl
    users = torch.randint(0, NU, (64,))
    pos = torch.randint(0, NI, (64,))
    neg = torch.randint(0, NI, (64,))
    torch.manual_seed(11)
    out = m(users, pos, neg)
    assert len(out) == 4, "the contrastive term must be returned apart from the regulariser"
    torch.manual_seed(11)
    u, i = m.user_emb.weight, m.item_emb.weight
    v1u, v1i = _ref_forward(adj, u, i, True)
    v2u, v2i = _ref_forward(adj, u, i, True)
    uidx, iidx = torch.unique(users), torch.unique(pos)
    ref = LAM * (_ref_infonce(v1u[uidx], v2u[uidx], TAU) + _ref_infonce(v1i[iidx], v2i[iidx], TAU))
    assert torch.allclose(out[3], ref, atol=1e-6)


def test_trainer_adds_the_contrastive_term_at_full_weight(simgcl):
    train = import_module("train")
    m, _ = simgcl
    users = torch.randint(0, NU, (64,))
    pos = torch.randint(0, NI, (64,))
    neg = torch.randint(0, NI, (64,))

    class OneBatch:
        def __iter__(self):
            yield users, pos, neg

        def __len__(self):
            return 1

    wd = 1e-5
    torch.manual_seed(11)
    o = m(users, pos, neg)
    expect = (train.bpr_loss(o[0], o[1]) + wd * o[2] + o[3]).item()
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    torch.manual_seed(11)
    got = train.train_epoch(m, OneBatch(), opt, wd, "cpu")
    assert abs(got - expect) < 1e-5


def _grad_delta(cls, **kw):
    """How much the contrastive weight moves the embedding gradient."""
    adj = _graph()
    grads = []
    for w in (0.0, 0.5):
        torch.manual_seed(1)
        m = cls(NU, NI, D, L, cl_weight=w, **kw)
        m.set_adj(adj)
        users = torch.randint(0, NU, (64,))
        pos = torch.randint(0, NI, (64,))
        neg = torch.randint(0, NI, (64,))
        torch.manual_seed(11)
        o = m(users, pos, neg)
        loss = -F.logsigmoid(o[0] - o[1]).mean() + 1e-5 * o[2] + (o[3] if len(o) > 3 else 0)
        grads.append(torch.autograd.grad(loss, m.user_emb.weight)[0])
    return (grads[1] - grads[0]).norm().item()


def test_simgcl_contrastive_term_moves_the_gradient():
    assert _grad_delta(import_module("models.simgcl").SimGCL) > 1e-3


def test_xsimgcl_contrastive_term_moves_the_gradient():
    assert _grad_delta(import_module("models.xsimgcl").XSimGCL) > 1e-3


def test_lightgcl_contrastive_term_moves_the_gradient():
    assert _grad_delta(import_module("models.lightgcl").LightGCL) > 1e-3


# ---------------------------------------------------------------- XSimGCL
X_EPS, X_LAM, X_TAU, X_LSTAR = 0.2, 0.2, 0.15, 1


def _ref_xsim_forward(adj, u, i, perturbed):
    """SELFRec XSimGCL_Encoder.forward, with .cuda() removed."""
    ego = torch.cat([u, i], 0)
    layers, cl = [], ego
    for k in range(L):
        ego = torch.sparse.mm(adj, ego)
        if perturbed:
            ego = ego + torch.sign(ego) * F.normalize(torch.rand_like(ego), dim=-1) * X_EPS
        layers.append(ego)
        if k == X_LSTAR - 1:
            cl = ego
    final = torch.stack(layers, dim=1).mean(dim=1)
    return final[:NU], final[NU:], cl[:NU], cl[NU:]


@pytest.fixture
def xsimgcl():
    XSimGCL = import_module("models.xsimgcl").XSimGCL
    adj = _graph()
    m = XSimGCL(NU, NI, D, L, layer_cl=X_LSTAR, noise_eps=X_EPS, cl_weight=X_LAM,
                cl_temp=X_TAU, cl_chunk=7)
    m.set_adj(adj)
    return m, adj


def test_xsimgcl_passes_equal_the_authors_encoder(xsimgcl):
    m, adj = xsimgcl
    u, i = m.user_emb.weight, m.item_emb.weight
    for perturbed, seed in ((False, 0), (True, 7)):
        torch.manual_seed(seed)
        ours = torch.cat(m._propagate(perturbed))
        torch.manual_seed(seed)
        ref = torch.cat(_ref_xsim_forward(adj, u, i, perturbed))
        assert torch.allclose(ours, ref, atol=1e-6)


def test_xsimgcl_one_pass_serves_both_losses_as_the_authors_write_it(xsimgcl):
    m, adj = xsimgcl
    users = torch.randint(0, NU, (64,))
    pos = torch.randint(0, NI, (64,))
    neg = torch.randint(0, NI, (64,))
    torch.manual_seed(11)
    out = m(users, pos, neg)
    assert len(out) == 4
    torch.manual_seed(11)
    fu, fi, cu, ci = _ref_xsim_forward(adj, m.user_emb.weight, m.item_emb.weight, True)
    # the ranking scores come from the SAME perturbed pass as the contrastive views
    assert torch.allclose(out[0], (fu[users] * fi[pos]).sum(1), atol=1e-6)
    assert torch.allclose(out[1], (fu[users] * fi[neg]).sum(1), atol=1e-6)
    uidx, iidx = torch.unique(users), torch.unique(pos)
    ref = X_LAM * (_ref_infonce(fu[uidx], cu[uidx], X_TAU) + _ref_infonce(fi[iidx], ci[iidx], X_TAU))
    assert torch.allclose(out[3], ref, atol=1e-6)


# ---------------------------------------------------------------- LightGCL
G_LAM, G_TAU = 0.2, 0.2


def _ref_lightgcl(m, rn, uids, iids, pos, neg):
    """HKUDS/LightGCL model.py, training branch, dropout 0, with the model's SVD factors."""
    u_mul_s = m.svd_u @ torch.diag(m.svd_s)
    v_mul_s = m.svd_v @ torch.diag(m.svd_s)
    ut, vt = m.svd_u.T, m.svd_v.T
    e_u, e_i = [m.user_emb.weight], [m.item_emb.weight]
    g_u, g_i = [m.user_emb.weight], [m.item_emb.weight]
    for layer in range(1, L + 1):
        z_u = rn @ e_i[layer - 1]
        z_i = rn.T @ e_u[layer - 1]
        g_u.append(u_mul_s @ (vt @ e_i[layer - 1]))
        g_i.append(v_mul_s @ (ut @ e_u[layer - 1]))
        e_u.append(z_u)
        e_i.append(z_i)
    G_u, G_i, E_u, E_i = sum(g_u), sum(g_i), sum(e_u), sum(e_i)
    neg_score = torch.log(torch.exp(G_u[uids] @ E_u.T / G_TAU).sum(1) + 1e-8).mean()
    neg_score += torch.log(torch.exp(G_i[iids] @ E_i.T / G_TAU).sum(1) + 1e-8).mean()
    pos_score = (torch.clamp((G_u[uids] * E_u[uids]).sum(1) / G_TAU, -5.0, 5.0)).mean() \
        + (torch.clamp((G_i[iids] * E_i[iids]).sum(1) / G_TAU, -5.0, 5.0)).mean()
    loss_s = -pos_score + neg_score
    return (E_u[uids] * E_i[pos]).sum(-1), (E_u[uids] * E_i[neg]).sum(-1), G_LAM * loss_s, E_u, E_i


@pytest.fixture
def lightgcl():
    LightGCL = import_module("models.lightgcl").LightGCL
    adj = _graph()
    m = LightGCL(NU, NI, D, L, svd_q=5, cl_weight=G_LAM, cl_temp=G_TAU, cl_chunk=7)
    m.set_adj(adj)
    return m, adj


def test_lightgcl_equals_the_authors_model(lightgcl):
    m, adj = lightgcl
    rn = adj.to_dense()[:NU, NU:]          # the normalised user-item block
    users = torch.randint(0, NU, (64,))
    pos = torch.randint(0, NI, (64,))
    neg = torch.randint(0, NI, (64,))
    out = m(users, pos, neg)
    assert len(out) == 4
    ref = _ref_lightgcl(m, rn, users, torch.cat([pos, neg]), pos, neg)
    assert torch.allclose(out[0], ref[0], atol=1e-5)
    assert torch.allclose(out[1], ref[1], atol=1e-5)
    assert torch.allclose(out[3], ref[2], atol=1e-5)
    # prediction scores come from the same summed main view
    assert torch.allclose(m.predict(torch.arange(NU)), ref[3] @ ref[4].T, atol=1e-5)


def test_lightgcl_chunked_negatives_equal_the_unchunked_value_and_gradient(lightgcl):
    m, _ = lightgcl
    g = torch.randn(33, D, requires_grad=True)
    e_all = torch.randn(50, D, requires_grad=True)
    rows = e_all[:33]
    ours = m._cl_side(g, rows, e_all)
    g_ours = torch.autograd.grad(ours, (g, e_all))
    ref = -torch.clamp((g * rows).sum(1) / G_TAU, -5.0, 5.0).mean() \
        + torch.log(torch.exp(g @ e_all.T / G_TAU).sum(1) + 1e-8).mean()
    g_ref = torch.autograd.grad(ref, (g, e_all))
    assert torch.allclose(ours, ref, atol=1e-5)
    assert torch.allclose(torch.cat([x.flatten() for x in g_ours]),
                          torch.cat([x.flatten() for x in g_ref]), atol=1e-5)


def test_lightgcl_svd_is_the_authors_call_drawn_from_the_run_seed():
    """The view is torch.svd_lowrank(block, q) with its default iterations, as in the
    authors' main.py, drawn from the seeded global generator: reproducible per seed."""
    LightGCL = import_module("models.lightgcl").LightGCL
    adj = _graph()

    def built(seed):
        torch.manual_seed(seed)
        m = LightGCL(NU, NI, D, L, svd_q=5)       # init draws first, as in a run
        m.set_adj(adj)
        return m

    a, b = built(3), built(3)
    for name in ("svd_u", "svd_s", "svd_v"):
        assert torch.equal(getattr(a, name), getattr(b, name))
    torch.manual_seed(3)
    LightGCL(NU, NI, D, L, svd_q=5)               # consume the same init draws
    U, S, V = torch.svd_lowrank(adj.to_dense()[:NU, NU:], q=5)
    assert torch.allclose(S, a.svd_s, atol=1e-5)
    assert torch.allclose((U * S) @ V.T, (a.svd_u * a.svd_s) @ a.svd_v.T, atol=1e-5)


def test_lightgcl_predicts_from_the_main_view_without_svd_factors(lightgcl):
    """Evaluation from a checkpoint builds LightGCL with only its adjacency (no set_adj, so no
    SVD factors). Prediction must work there and equal what the trained model scores."""
    full, adj = lightgcl
    bare = import_module("models.lightgcl").LightGCL(NU, NI, D, L)
    bare.load_state_dict(full.state_dict())
    bare.norm_adj = adj
    users = torch.arange(NU)
    e_u, e_i, _, _ = full._propagate()
    assert torch.allclose(full.predict(users), e_u @ e_i.T, atol=1e-6)
    assert torch.allclose(bare.predict(users), full.predict(users), atol=1e-6)


# ------------------------------------------------ init option, flags, provenance
@pytest.mark.parametrize("modname,clsname", [("models.simgcl", "SimGCL"),
                                             ("models.xsimgcl", "XSimGCL"),
                                             ("models.lightgcl", "LightGCL")])
def test_uniform_init_is_xavier_uniform_and_recorded(modname, clsname):
    cls = getattr(import_module(modname), clsname)
    m = cls(NU, NI, D, L, init="uniform")
    bound = (6.0 / (NU + D)) ** 0.5
    w = m.user_emb.weight
    assert float(w.abs().max()) <= bound + 1e-6
    assert float(w.abs().max()) > 0.8 * bound          # spread out to the bound, not Gaussian-clustered
    assert m.hparams["init"] == "uniform"
    assert cls(NU, NI, D, L).hparams["init"] == "normal"  # the default is unchanged
    with pytest.raises(ValueError):
        cls(NU, NI, D, L, init="gaussian")


def test_build_model_refuses_flags_that_do_not_apply():
    run = import_module("run_experiment")
    adj = _graph()
    with pytest.raises(ValueError):
        run.build_model("lightgcn", NU, NI, 0, adj, overrides={"cl_weight": 0.1})
    with pytest.raises(ValueError):
        run.build_model("lightgcl", NU, NI, 0, adj, overrides={"noise_eps": 0.1})
    m = run.build_model("xsimgcl", NU, NI, 0, adj,
                        overrides={"cl_weight": 0.05, "noise_eps": 0.1, "cl_temp": 0.2,
                                   "layer_cl": 2, "init": "uniform", "unused": None})
    assert m.hparams == {"model": "xsimgcl", "cl_weight": 0.05, "noise_eps": 0.1,
                         "cl_temp": 0.2, "layer_cl": 2, "init": "uniform"}


def _grid():
    from pathlib import Path as P
    import importlib.util
    here = P(__file__).resolve().parents[1] / "scripts" / "contrastive_grid.py"
    if not here.exists():
        pytest.skip("scripts/contrastive_grid.py not present")
    spec = importlib.util.spec_from_file_location("_contrastive_grid", here)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _result(g, out, ckpt, cfg, hp, seed, val, **change):
    """Write a finished result that is a valid run of the protocol, then apply `change`
    ("section__field": value, or "section": value). Selection runs (seed 42) carry no test
    metrics; reported runs do."""
    selection = seed == g.SELECT_SEED
    _, ckpt_dir, result, log = g.paths(out, ckpt, cfg, hp, seed)
    mh = {"model": g.MODEL[cfg], "init": "uniform"}
    mh.update({g.HPARAM_KEY.get(k, k): v for k, v in g.canon(hp).items()})
    rec = {"best_val_metrics": {"NDCG@10": val},
           "test_metrics": None if selection else {"NDCG@10": val - 0.001},
           "best_epoch": 60, "total_epochs_trained": 80, "stopped_early": True,
           "config": dict(g.protocol(), patience_unit="epochs"),
           "data": dict(g.ML20M), "model_hparams": mh}
    for key, value in change.items():
        section, _, field = key.partition("__")
        if field:
            rec[section][field] = value
        else:
            rec[section] = value
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(rec))
    weights = ckpt_dir / cfg / g.ENCODER / f"seed-{seed}" / "best_model.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(f"{cfg}-{seed}-{sorted(hp.items())}".encode())
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("# checkout: abc123\n# env: test\n$ python3 run_experiment.py\n")
    return result


CENTRE = {"cl_weight": 0.1, "cl_temp": 0.2, "layer_cl": 2, "cl_eps": 0.1}


def _score(hp):
    """A smooth landscape with an interior optimum in every searched dimension."""
    import math
    return 0.07 - 0.001 * sum(abs(math.log(v / CENTRE[k])) for k, v in hp.items() if k in CENTRE)


def _freeze_all(g, out, ckpt, score=_score):
    """Fabricate every selection run `pick` asks for until all three choices freeze."""
    for _ in range(30):
        assert g.main(["pick", "--out", str(out), "--ckpt", str(ckpt)]) == 0
        st = g.load_state(out)
        if all(c in st["chosen"] for c in g.SEARCH):
            return st
        for r in st["select"]:
            if not g.paths(out, ckpt, r["cfg"], r["hp"], g.SELECT_SEED)[2].exists():
                _result(g, out, ckpt, r["cfg"], r["hp"], g.SELECT_SEED, score(r["hp"]))
    raise AssertionError("the searches did not freeze")


def test_grid_numbers_the_runs_before_any_result():
    g = _grid()
    st = g.fresh_state()
    assert [r["n"] for r in st["select"]] == list(range(1, 11))
    assert [r["cfg"] for r in st["select"]] == ["m1b"] * 4 + ["m1c"] * 3 + ["m1d"] * 3
    assert [r["hp"]["cl_weight"] for r in st["select"][:4]] == [0.05, 0.1, 0.2, 0.5]
    assert all(r["hp"]["layer_cl"] == 1 and r["hp"]["cl_eps"] == 0.1 for r in st["select"][4:7])
    assert [r["hp"]["cl_temp"] for r in st["select"][7:]] == [0.1, 0.2, 0.5]
    rep = g.reported_runs(st)
    assert sorted(rep) == list(range(101, 113)) and all(hp is None for _, hp, _ in rep.values())
    assert [rep[n][2] for n in range(101, 105)] == [123, 456, 789, 2026]
    # seed 42 selects, and the chosen setting's selection run is its seed-42 reported run
    assert g.SELECT_SEED == 42 and g.SELECT_SEED in g.REPORT_SEEDS
    assert g.NEW_SEEDS == (123, 456, 789, 2026)


def _walk(g, cfg, vals):
    """advance() over a dict {tuple(sorted hp items): val}."""
    return g.advance(cfg, lambda hp: vals.get(tuple(sorted(hp.items()))))


def _key(g, cfg, **v):
    spec = g.SEARCH[cfg]
    return tuple(sorted(g.canon({**spec["fixed"], **spec["start"], **v}).items()))


def test_grid_simgcl_lambda_extends_at_either_edge_and_breaks_ties_low():
    g = _grid()
    k = lambda lam: _key(g, "m1b", cl_weight=lam)
    interior = {k(0.05): .060, k(0.1): .062, k(0.2): .061, k(0.5): .050}
    status, hp, _ = _walk(g, "m1b", interior)
    assert status == "chosen" and hp["cl_weight"] == 0.1
    low = {k(0.05): .064, k(0.1): .062, k(0.2): .061, k(0.5): .050}
    status, need, _ = _walk(g, "m1b", low)
    assert status == "waiting" and [h["cl_weight"] for h in need] == [0.02]
    low[k(0.02)] = .065
    status, need, _ = _walk(g, "m1b", low)
    assert status == "waiting" and [h["cl_weight"] for h in need] == [0.01]
    low[k(0.01)] = .066
    status, hp, trace = _walk(g, "m1b", low)
    assert status == "chosen" and hp["cl_weight"] == 0.01 and "not bracketed" in trace[-1]
    high = {k(0.05): .050, k(0.1): .055, k(0.2): .060, k(0.5): .065}
    status, need, _ = _walk(g, "m1b", high)
    assert [h["cl_weight"] for h in need] == [1.0]
    high[k(1.0)] = .060
    status, hp, _ = _walk(g, "m1b", high)
    assert status == "chosen" and hp["cl_weight"] == 0.5
    tie = {k(0.05): .060, k(0.1): .062, k(0.2): .062, k(0.5): .050}
    assert _walk(g, "m1b", tie)[1]["cl_weight"] == 0.1


def test_grid_xsimgcl_follows_the_authors_order_lambda_then_lstar_then_eps():
    g = _grid()
    vals = {_key(g, "m1c", cl_weight=w): s for w, s in ((0.05, .060), (0.1, .063), (0.2, .061))}
    status, need, _ = _walk(g, "m1c", vals)
    assert status == "waiting"
    assert [(h["cl_weight"], h["layer_cl"]) for h in need] == [(0.1, 2), (0.1, 3)]
    vals[_key(g, "m1c", cl_weight=0.1, layer_cl=2)] = .062
    vals[_key(g, "m1c", cl_weight=0.1, layer_cl=3)] = .064
    status, need, _ = _walk(g, "m1c", vals)
    assert [(h["layer_cl"], h["cl_eps"]) for h in need] == [(3, 0.05), (3, 0.2)]
    vals[_key(g, "m1c", cl_weight=0.1, layer_cl=3, cl_eps=0.05)] = .063
    vals[_key(g, "m1c", cl_weight=0.1, layer_cl=3, cl_eps=0.2)] = .062
    status, hp, trace = _walk(g, "m1c", vals)
    assert status == "chosen" and hp == g.canon(
        {"cl_weight": 0.1, "cl_eps": 0.1, "cl_temp": 0.2, "layer_cl": 3})
    assert "at the end of the searched range" in trace[1]        # l* = 3 is the last layer


def test_grid_lightgcl_tau_extends_upward_only():
    g = _grid()
    k = lambda tau: _key(g, "m1d", cl_temp=tau)
    up = {k(0.1): .050, k(0.2): .055, k(0.5): .060}
    status, need, _ = _walk(g, "m1d", up)
    assert status == "waiting" and [h["cl_temp"] for h in need] == [1.0]
    down = {k(0.1): .060, k(0.2): .055, k(0.5): .050}
    status, hp, trace = _walk(g, "m1d", down)
    assert status == "chosen" and hp["cl_temp"] == 0.1 and "searched range" in trace[-1]


def test_grid_pick_registers_extensions_then_freezes_and_never_revisits(tmp_path):
    g = _grid()
    out, ckpt = tmp_path / "grid", tmp_path / "ckpt"
    base = [r["hp"] for r in g.fresh_state()["select"] if r["cfg"] == "m1d"]
    for hp, v in zip(base, (.050, .055, .060)):                 # tau 0.5 wins, at the edge
        _result(g, out, ckpt, "m1d", hp, g.SELECT_SEED, v)
    assert g.main(["pick", "--models", "m1d", "--out", str(out), "--ckpt", str(ckpt)]) == 0
    st = g.load_state(out)
    assert "m1d" not in st["chosen"]
    ext = [r for r in st["select"] if r["n"] == 11]
    assert ext and ext[0]["hp"]["cl_temp"] == 1.0 and "extending" in ext[0]["why"]
    _result(g, out, ckpt, "m1d", ext[0]["hp"], g.SELECT_SEED, .058)
    assert g.main(["pick", "--models", "m1d", "--out", str(out), "--ckpt", str(ckpt)]) == 0
    assert g.load_state(out)["chosen"]["m1d"]["hp"]["cl_temp"] == 0.5
    _result(g, out, ckpt, "m1d", ext[0]["hp"], g.SELECT_SEED, .070)   # results change later
    assert g.main(["pick", "--models", "m1d", "--out", str(out), "--ckpt", str(ckpt)]) == 0
    assert g.load_state(out)["chosen"]["m1d"]["hp"]["cl_temp"] == 0.5  # frozen stays frozen


def test_grid_nothing_touches_test_until_every_choice_is_frozen(tmp_path):
    g = _grid()
    out, ckpt = tmp_path / "grid", tmp_path / "ckpt"
    base = [r["hp"] for r in g.fresh_state()["select"] if r["cfg"] == "m1d"]
    for hp, v in zip(base, (.050, .062, .055)):
        _result(g, out, ckpt, "m1d", hp, g.SELECT_SEED, v)
    assert g.main(["pick", "--models", "m1d", "--out", str(out), "--ckpt", str(ckpt)]) == 0
    common = ["--out", str(out), "--ckpt", str(ckpt)]
    assert g.main(["seeds", "--only", "111", "--dry-run"] + common) == 2
    assert g.main(["grid-test"] + common) == 2
    assert g.main(["collect"] + common) == 2


def test_grid_commands_carry_the_protocol(tmp_path, capsys):
    g = _grid()
    out, ckpt = tmp_path / "grid", tmp_path / "ckpt"
    assert g.main(["select", "--smoke", "--only", "1", "--out", str(out), "--ckpt", str(ckpt),
                   "--dry-run"]) == 0
    cmd = [ln for ln in capsys.readouterr().out.splitlines() if "run_experiment.py" in ln][0]
    for flag in ("--config m1b", "--seed 42 ", "--no-test", "--min-epochs 45",
                 "--dense-eval-until 20", "--init uniform", "--cl-weight 0.05", "--cl-eps 0.1",
                 "--cl-temp 0.2"):
        assert flag in cmd + " ", flag
    _freeze_all(g, out, ckpt)
    capsys.readouterr()
    assert g.main(["seeds", "--smoke", "--only", "101", "--out", str(out), "--ckpt", str(ckpt),
                   "--dry-run"]) == 0
    cmd = [ln for ln in capsys.readouterr().out.splitlines() if "run_experiment.py" in ln][0]
    assert "--seed 123 " in cmd + " " and "--no-test" not in cmd and "--min-epochs 45" in cmd


@pytest.mark.parametrize("change", [{"config__num_epochs": 1}, {"config__batch_size": 4096},
                                    {"config__min_epochs": 0}, {"config__dense_eval_until": 0},
                                    {"data__n_users": 11000}, {"model_hparams__cl_temp": 0.3},
                                    {"model_hparams__init": "normal"},
                                    {"test_metrics": {"NDCG@10": 0.06}}])
def test_grid_pick_refuses_a_result_that_is_not_a_protocol_run(tmp_path, change):
    g = _grid()
    out, ckpt = tmp_path / "grid", tmp_path / "ckpt"
    base = [r["hp"] for r in g.fresh_state()["select"] if r["cfg"] == "m1d"]
    for i, hp in enumerate(base):
        _result(g, out, ckpt, "m1d", hp, g.SELECT_SEED, .05 + .005 * (i == 1),
                **(change if i == 1 else {}))
    common = ["--models", "m1d", "--out", str(out), "--ckpt", str(ckpt)]
    assert g.main(["pick"] + common) == 2
    assert "m1d" not in g.load_state(out)["chosen"]
    assert g.main(["pick", "--smoke"] + common) == 0            # trials only


def test_grid_will_not_skip_a_trial_result_as_done(tmp_path, capsys):
    g = _grid()
    out, ckpt = tmp_path / "grid", tmp_path / "ckpt"
    data = tmp_path / "ml20m"
    data.mkdir()
    (data / "stats.json").write_text(json.dumps(g.ML20M))
    hp = g.fresh_state()["select"][7]["hp"]                        # run #8
    _result(g, out, ckpt, "m1d", hp, g.SELECT_SEED, .03, config__num_epochs=1)
    args = ["select", "--models", "m1d", "--out", str(out), "--ckpt", str(ckpt), "--dry-run",
            "--", "--data-dir", str(data)]
    assert g.main(args) == 2
    assert "not protocol runs" in capsys.readouterr().err
    _result(g, out, ckpt, "m1d", hp, g.SELECT_SEED, .03)          # a real one is skipped
    assert g.main(args) == 0
    text = capsys.readouterr().out
    assert "skip" in text and text.count("run_experiment.py") == 2


def test_grid_refuses_to_launch_on_a_split_that_is_not_ml20m(tmp_path, capsys):
    g = _grid()
    out, ckpt = tmp_path / "grid", tmp_path / "ckpt"
    amazon = tmp_path / "amazon"
    amazon.mkdir()
    (amazon / "stats.json").write_text(json.dumps({"n_users": 11000, "n_items": 9332}))
    args = ["select", "--only", "8", "--out", str(out), "--ckpt", str(ckpt), "--dry-run",
            "--", "--data-dir", str(amazon)]
    assert g.main(args) == 2
    assert "is not the ML-20M split" in capsys.readouterr().err
    assert g.main(["select", "--smoke"] + args[1:]) == 0
    assert g.main(args[:-1] + [str(tmp_path / "missing")]) == 2


@pytest.mark.parametrize("args", [["seeds", "--only", "3"], ["select", "--only", "101"],
                                  ["seeds", "--only", "113"],
                                  ["select", "--only", "12"], ["select", "--only", "x"],
                                  ["select", "--only", "1", "--models", "m1b"],
                                  ["plan", "--only", "1"], ["select", "--only", ","]])
def test_grid_only_refuses_a_number_that_does_not_fit(tmp_path, args):
    g = _grid()
    assert g.main(args + ["--smoke", "--out", str(tmp_path / "o"), "--ckpt", str(tmp_path / "c"),
                          "--dry-run"]) == 2


def test_grid_anchors_a_relative_data_dir(tmp_path, capsys, monkeypatch):
    g = _grid()
    monkeypatch.chdir(tmp_path)
    base = ["select", "--smoke", "--only", "8", "--out", str(tmp_path / "o"),
            "--ckpt", str(tmp_path / "c"), "--dry-run", "--"]
    for spelling in (["--data-dir", "rel/splits"], ["--data-dir=rel/splits"]):
        assert g.main(base + spelling) == 0
        cmds = [ln for ln in capsys.readouterr().out.splitlines() if "run_experiment.py" in ln]
        assert cmds and all(ln.endswith(f"--data-dir {tmp_path / 'rel' / 'splits'}") for ln in cmds)


def _grade_selection_runs(g, out, ckpt, st, tamper=None):
    """Stand in for grid-test: write grid_test.json for every finished selection run."""
    import hashlib
    rows = []
    for r in st["select"]:
        _, ckpt_dir, _, _ = g.paths(out, ckpt, r["cfg"], r["hp"], g.SELECT_SEED)
        w = ckpt_dir / r["cfg"] / g.ENCODER / f"seed-{g.SELECT_SEED}" / "best_model.pt"
        if w.is_file():
            digest = hashlib.sha256(w.read_bytes()).hexdigest()
            rows.append({"n": r["n"], "cfg": r["cfg"], "hp": r["hp"],
                         "best_model_sha256": "0" * 64 if r["n"] == tamper else digest,
                         "test_metrics": {"NDCG@10": 0.0691, "MRR": 0.2}})
    (out / "grid_test.json").write_text(json.dumps({"runs": rows}))


def test_grid_collect_takes_seed_42_from_the_graded_selection_run(tmp_path):
    import hashlib
    g = _grid()
    out, ckpt = tmp_path / "grid", tmp_path / "ckpt"
    st = _freeze_all(g, out, ckpt)
    common = ["--out", str(out), "--ckpt", str(ckpt)]
    for n, (cfg, hp, s) in g.reported_runs(st).items():
        if n != 112:
            _result(g, out, ckpt, cfg, hp, s, .07)
    assert g.main(["collect"] + common) == 2                     # no grid-test yet
    _grade_selection_runs(g, out, ckpt, st)
    assert g.main(["collect"] + common) == 2                     # #112 is not finished
    cfg, hp, s = g.reported_runs(st)[112]
    _result(g, out, ckpt, cfg, hp, s, .07)
    chosen_m1b = g.find_run(st, "m1b", st["chosen"]["m1b"]["hp"])["n"]
    _grade_selection_runs(g, out, ckpt, st, tamper=chosen_m1b)
    assert g.main(["collect"] + common) == 2                     # checkpoint differs from graded
    _grade_selection_runs(g, out, ckpt, st)
    assert g.main(["collect"] + common) == 0
    final = out / "final"
    man = json.loads((final / "MANIFEST.json").read_text())
    assert sorted(man["configs"]) == ["m1b", "m1c", "m1d"] and "sensitivity" in man
    assert man["chosen_sha256"] == g.load_state(out)["chosen_sha256"]
    for cfg, entry in man["configs"].items():
        assert sorted(entry["seeds"], key=int) == [str(x) for x in g.REPORT_SEEDS]
        s42 = entry["seeds"]["42"]
        assert s42["selection_run"] is True and s42["test_ndcg10"] == 0.0691
        assert s42["run"] == g.find_run(st, cfg, entry["hparams"])["n"]
        assert not any(v["selection_run"] for k, v in entry["seeds"].items() if k != "42")
        for s, row in entry["seeds"].items():
            w = final / "checkpoints" / cfg / g.ENCODER / f"seed-{s}" / "best_model.pt"
            assert row["best_model_sha256"] == hashlib.sha256(w.read_bytes()).hexdigest()
            assert row["trained_from"] == ["abc123"] and row["environment"] == ["test"]
        leaf = json.loads((final / "results" / cfg / g.ENCODER / "seed-42" / "results.json").read_text())
        assert leaf["test_metrics"]["NDCG@10"] == 0.0691 and "_provenance" in leaf
        assert entry["search"]["hp"] == entry["hparams"]
    assert (final / "search.json").is_file() and (final / "grid_test.json").is_file()
    assert g.main(["collect"] + common) == 0                     # collecting again is safe
    assert g.main(["collect", "--smoke"] + common) == 2          # trial and real never mix
    foreign = tmp_path / "canonical"
    foreign.mkdir()
    (foreign / "keep.json").write_text("{}")
    assert g.main(["collect", "--final", str(foreign)] + common) == 2
    assert sorted(x.name for x in foreign.iterdir()) == ["keep.json"]


def _dead_pid():
    import subprocess
    import sys
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_grid_lock_is_taken_released_and_taken_over_when_stale(tmp_path):
    import os
    import socket
    g = _grid()
    lock = tmp_path / "logs" / "run.lock"
    assert g.acquire(lock) is None
    assert str(os.getpid()) in lock.read_text()
    assert g.acquire(lock) is not None
    lock.write_text(f"{socket.gethostname()} {_dead_pid()}\n")
    assert g.lock_holder(lock) is None and g.acquire(lock) is None
    lock.write_text("other-host 1\n")
    assert "cannot be checked from here" in g.lock_holder(lock)


def test_grid_refuses_a_run_that_is_already_going(tmp_path, capsys):
    import os
    import socket
    g = _grid()
    out, ckpt = tmp_path / "grid", tmp_path / "ckpt"
    hp = g.fresh_state()["select"][0]["hp"]
    _, _, _, log = g.paths(out, ckpt, "m1b", hp, g.SELECT_SEED)
    lock = g.lock_path(log)
    lock.parent.mkdir(parents=True)
    args = ["select", "--smoke", "--only", "1", "--out", str(out), "--ckpt", str(ckpt), "--dry-run"]
    lock.write_text(f"{socket.gethostname()} {os.getpid()}\n")
    assert g.main(args) == 2
    assert "already going" in capsys.readouterr().err
    lock.write_text(f"{socket.gethostname()} {_dead_pid()}\n")
    assert g.main(args) == 0


def test_grid_run_queue_locks_each_run_and_skips_a_busy_one(tmp_path):
    import os
    import socket
    import sys
    g = _grid()
    out = tmp_path / "grid"
    log = out / "logs" / "a_seed0.log"
    marker = tmp_path / "ran"
    cmd = [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('x')"]
    assert g.run_queue([(cmd, log, "a")], [], out, "abc123") == ([], [])
    assert marker.exists() and not g.lock_path(log).exists()
    text = log.read_text()
    assert text.startswith("# checkout: abc123") and "# env: python" in text
    marker.unlink()
    g.lock_path(log).write_text(f"{socket.gethostname()} {os.getpid()}\n")
    assert g.run_queue([(cmd, log, "a")], [], out, "abc123") == ([], ["a"])
    assert not marker.exists()


def test_grid_self_check_passes_on_this_checkout():
    assert _grid().self_check() == []
