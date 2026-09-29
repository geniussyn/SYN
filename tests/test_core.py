"""Numerical core, checked against direct definitions on synthetic data."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from syn.exact import neighbourhoods, topk          # noqa: E402
from syn.pull import active_pairs                   # noqa: E402
from syn.train import Regions, qradial              # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def unit(rng, n, d, dtype=np.float16):
    X = rng.standard_normal((n, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(dtype)


def test_topk_matches_brute_force():
    rng = np.random.default_rng(0)
    X, Q = unit(rng, 4000, 32), unit(rng, 37, 32)
    idx, sim = topk(Q, X, 12, qchunk=8, dchunk=997, device=DEV)
    ref = np.asarray(Q, np.float32) @ np.asarray(X, np.float32).T
    exp = np.argsort(-ref, axis=1, kind="stable")[:, :12]
    assert (idx == exp).all()
    assert np.allclose(sim, np.take_along_axis(ref, exp, 1), atol=1e-5)


def test_neighbourhoods_exclude_self_and_are_rank_1_based():
    rng = np.random.default_rng(1)
    X = unit(rng, 2000, 24)
    ids = np.array([3, 17, 900, 1999])
    L = 25
    NL = neighbourhoods(ids, X, L, qchunk=2, device=DEV)
    ref = np.asarray(X, np.float32) @ np.asarray(X, np.float32).T
    for r, u in enumerate(ids):
        assert u not in NL[r]
        order = np.argsort(-ref[u], kind="stable")
        assert (NL[r] == order[order != u][:L]).all()
        # v in N_L(u)  <=>  rank_u(v) <= L, with rank counting u itself
        v = int(NL[r][L - 1])
        assert int((ref[u] > ref[u, v]).sum()) == L


def test_active_pairs_matches_definition():
    rng = np.random.default_rng(2)
    X = unit(rng, 3000, 16)
    S = rng.choice(3000, (12, 6), replace=False).astype(np.int32)
    anchors = np.unique(S)
    L = 40
    NL = neighbourhoods(anchors, X, L, qchunk=16, device=DEV)
    pq, pi, pj = active_pairs(S, NL, anchors)
    got = {(int(q), int(i), int(j)) for q, i, j in zip(pq, pi, pj)}
    row = {int(d): r for r, d in enumerate(anchors)}
    want = set()
    for q in range(S.shape[0]):
        for i in range(S.shape[1]):
            for j in range(i + 1, S.shape[1]):
                u, v = int(S[q, i]), int(S[q, j])
                if v not in set(NL[row[u]]) and u not in set(NL[row[v]]):
                    want.add((q, i, j))
    assert got == want and all(i < j for i, j in zip(pi, pj))


def test_region_sampling_respects_pool_and_forbid():
    rng = np.random.default_rng(3)
    assign = rng.integers(0, 8, 500).astype(np.int32)
    reg = Regions(assign, 8)
    for v in (0, 42, 499):
        members = set(np.flatnonzero(assign == assign[v]).tolist())
        forbid = {v, int(rng.choice(sorted(members)))}
        got = reg.sample(v, 4, forbid, rng)
        assert len(got) == 4 and len(set(got.tolist())) == 4
        assert set(got.tolist()) <= members - forbid


def test_pull_loss_matches_equation():
    rng = np.random.default_rng(4)
    V = torch.tensor(unit(rng, 20, 8, np.float32), device=DEV)
    di, dj, bi, bj = [0, 2], [1, 3], [10, 12], [11, 13]
    m, tau = 0.0, 0.05
    s_ij = (V[di] * V[dj]).sum(-1)
    got = 0.5 * (F.softplus(((V[di] * V[bi]).sum(-1) - s_ij + m) / tau)
                 + F.softplus(((V[dj] * V[bj]).sum(-1) - s_ij + m) / tau))
    want = torch.tensor([
        0.5 * (F.softplus((V[a] @ V[c] - V[a] @ V[b] + m) / tau)
               + F.softplus((V[b] @ V[d] - V[b] @ V[a] + m) / tau)).item()
        for a, b, c, d in zip(di, dj, bi, bj)], device=DEV)
    assert torch.allclose(got, want, atol=1e-5)


def test_plain_pull_gradient_uses_the_whole_direction():
    """The `plain` ablation must not be confined to the query-radial tangent."""
    rng = np.random.default_rng(5)
    V = torch.tensor(unit(rng, 4, 8, np.float32), device=DEV, requires_grad=True)
    q = torch.tensor(unit(rng, 1, 8, np.float32), device=DEV)[0]
    s_ij = (V[0] * V[1]).sum()
    F.softplus(((V[0] * V[2]).sum() - s_ij) / 0.05).backward()
    g = V.grad[0]
    t = q - (q @ V[0].detach()) * V[0].detach()
    t = t / t.norm()
    assert (g - (g @ t) * t).norm() > 0.1 * g.norm()


def test_qradial_is_the_identity_forward_and_a_projector_backward():
    """SYN's pull gradient. The forward pass must leave the loss untouched and
    the Jacobian must be exactly t t^T, which fails if the tangent is left
    attached to the graph."""
    rng = np.random.default_rng(6)
    q = torch.tensor(unit(rng, 5, 16, np.float32), device=DEV)
    z0 = torch.tensor(unit(rng, 5, 16, np.float32), device=DEV)
    z = z0.clone().requires_grad_(True)
    out = qradial(z, q)
    assert torch.allclose(out, z0, atol=1e-6), "forward is not the identity"
    g = torch.randn(5, 16, device=DEV)
    out.backward(g)
    with torch.no_grad():
        a = (z0 * q).sum(-1, keepdim=True)
        rho = (1 - a * a).clamp(min=1e-12).sqrt()
        u = (z0 - a * q) / rho
        t = rho * q - a * u
        t = t / t.norm(dim=-1, keepdim=True)
        want = (g * t).sum(-1, keepdim=True) * t
        assert torch.allclose(z.grad, want, atol=1e-6), "Jacobian is not t t^T"


def test_qradial_detaches_the_query():
    """The query is held fixed through the operator (sg[q] in Appendix A)."""
    rng = np.random.default_rng(7)
    z = torch.tensor(unit(rng, 3, 12, np.float32), device=DEV)
    q = torch.tensor(unit(rng, 3, 12, np.float32), device=DEV, requires_grad=True)
    qradial(z.clone().requires_grad_(True), q).sum().backward()
    assert q.grad is None or float(q.grad.abs().sum()) == 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
