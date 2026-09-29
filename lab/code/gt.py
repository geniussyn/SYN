"""Exact top-k of every eval query over the corpus, on GPU. Ruler #1 material.

ACCUMULATED IN FP32, NOT FP16. The vectors are stored fp16 and hnswlib loads an
fp32 copy of exactly those bits, so the reference an index should be scored
against is the fp32 exhaustive search over the same data. Accumulating the dot
products in fp16 instead is a lower-precision approximation, and where several
documents sit within a half-precision ulp of each other it orders them
differently from the index -- which then loses recall for arithmetic reasons
while the search did nothing wrong.

That was not hypothetical: under fp16 the MS MARCO SYN arm's exact top-10
disagreed with the fp32 one on 3.2% of its hard decile, producing a recall
ceiling at 0.9683 that no search budget moved and that looked, wrongly, like
unreachable documents. Vanilla lost 0.36% the same way, so the error was also
asymmetric between the arms. SYN compresses the score distribution, which is
what exposes it.
"""
import argparse, json
from pathlib import Path
import numpy as np, torch

from paths import EMB, ROOT as LAB                          # noqa: E402
RES = LAB / "results"

def exact_topk(Q, X, k, qchunk=512, dchunk=1_000_000, device="cuda"):
    """Q (nq,d) fp16 np, X (n,d) fp16 np (may be memmap). Returns (idx, sim) top-k by IP.

    fp32 throughout: this is the ruler, and it has to be at least as precise as
    the thing it measures. See the module docstring."""
    nq, n = Q.shape[0], X.shape[0]
    Qg = torch.from_numpy(np.ascontiguousarray(Q)).to(device).float()
    best_v = torch.full((nq, k), -3.4e38, device=device, dtype=torch.float32)
    best_i = torch.zeros((nq, k), device=device, dtype=torch.int64)
    for s in range(0, n, dchunk):
        e = min(s + dchunk, n)
        Xg = torch.from_numpy(np.ascontiguousarray(X[s:e])).to(device).float()
        for qs in range(0, nq, qchunk):
            qe = min(qs + qchunk, nq)
            sim = Qg[qs:qe] @ Xg.T
            v, i = torch.topk(sim, min(k, e - s), dim=1)
            cv = torch.cat([best_v[qs:qe], v], 1)
            ci = torch.cat([best_i[qs:qe], i + s], 1)
            v2, j = torch.topk(cv, k, dim=1)
            best_v[qs:qe] = v2
            best_i[qs:qe] = ci.gather(1, j)
        del Xg
        torch.cuda.empty_cache()
    return best_i.cpu().numpy().astype(np.int32), best_v.cpu().numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True); ap.add_argument("--arm", required=True)
    ap.add_argument("-k", type=int, default=10)
    a = ap.parse_args()
    base = EMB / f"{a.corpus}__{a.arm}"
    X = np.load(str(base) + "_docs.npy", mmap_mode="r")
    Q = np.load(str(base) + "_queries.npy")
    idx, sim = exact_topk(Q, X, a.k)
    RES.mkdir(exist_ok=True, parents=True)
    out = RES / f"gt_{a.corpus}__{a.arm}_k{a.k}.npz"
    np.savez_compressed(out, idx=idx, sim=sim)
    print(f"wrote {out}  {idx.shape}", flush=True)

if __name__ == "__main__":
    main()
