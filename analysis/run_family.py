"""Index families and libraries on one embedding, at matched build threads.

Search is timed single-threaded in C++ for every family: cpp/hnsw_bench for
hnswlib, cpp/faiss_bench for faiss (HNSW, NSG, IVF), DiskANN's own binary for
Vamana. Construction is not the measured quantity but is reported, so the build
cost of each library is comparable: BUILD_THREADS is the same for all.

IVF is not a graph index. It is here because the paper's claim is specifically
about graph indexes built from document-document proximity; a partition index
is the contrast case.
"""
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import EMB, ROOT  # noqa: E402
# LAB comes from paths.ROOT: SYN_ROOT if set, else this file's parent.
# It used to be a fixed absolute path, which resolved only on the machine
# this was written on, where that path was a symlink into the tree.
LAB = ROOT; RES = LAB / "results"
WORK = Path(os.environ.get("SYN_INDEX_DIR", "/dev/shm/synidx"))
DISKANN = LAB / "work/DiskANN/build/apps"
N_THREADS = int(os.environ.get("SYN_THREADS", "1"))               # search: one core
BUILD_THREADS = int(os.environ.get("SYN_BUILD_THREADS", "100"))   # build: matched
from metrics import recall_at_k, mrr_at_k, ndcg_at_k, \
    qrels_recall_at_k                                   # noqa: E402
from run_cell import load_rels                          # noqa: E402
from index_hnsw import build as hnswlib_build, write_fbin, timed_grid, \
    timing_lock, PIN_CORE, EXCLUSIVE                             # noqa: E402

EF_GRID = [16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224,
           256, 320, 384, 448, 512]
L_GRID = [16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224,
           256, 320, 384, 448, 512]
NPROBE = [1, 2, 3, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64, 80, 96, 128,
          160, 192, 256, 320, 384, 448, 512]
# Deep points, timed on the hard decile only and stopped by the latency budget.
# The shallow grid leaves the baseline unmeasured where the comparison is: on
# IVF it reached 0.804 and stopped, so every target above that was a gap in the
# measurement rather than a result. nprobe is capped at nlist (a full scan).
DEEP = [768, 1024, 1536, 2048, 3072, 4096]
# At or above this budget one timed pass already runs far past the harness's
# three-second floor, so the min-of-three that protects a short run from
# scheduling noise buys nothing and costs three times the wall clock. Below it,
# three passes as before. The realised count is recorded per row.
DEEP_REPS_FROM = 64
# Recorded next to every point, never used to stop one: a target is measured
# until it is reached, and the only ceiling is the parameter's own (nprobe =
# nlist is a full scan; a graph's ef has nothing left to widen into).
MS_NOTE = 64.0
EF_MAX = 131072
TARGETS = [0.90, 0.95, 0.99]
FAMILIES = {
    "hnswlib":    ("HNSW (hnswlib)",  "graph"),
    "faiss-hnsw": ("HNSW (faiss)",    "graph"),
    "faiss-nsg":  ("NSG (faiss)",     "graph"),
    "vamana":     ("Vamana (DiskANN)", "graph"),
    "faiss-ivf":  ("IVF (faiss)",     "partition"),
}


def faiss_bench(index_path, Q, params, k, tag, reps=3):
    qp = WORK / f"{tag}_q.fbin"; pref = WORK / f"{tag}_res"
    write_fbin(qp, Q)
    env = {**os.environ, "OMP_PROC_BIND": "true", "OMP_PLACES": "cores"}
    pre = ["taskset", "-c", str(PIN_CORE)] if EXCLUSIVE else []
    with timing_lock():
        r = subprocess.run(pre + [str(LAB / "cpp/faiss_bench"),
                                  str(index_path), str(qp), str(k), str(N_THREADS),
                                  str(reps), str(pref), ",".join(map(str, params)), "3.0"],
                           capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise SystemExit(f"faiss_bench failed: {r.stderr[-2000:]}")
    rows = []
    for row in json.loads(r.stdout):
        p, w = row["param"], row["wall_s"]
        lp = Path(str(pref) + f"_p{p}.u32")
        I = np.fromfile(lp, dtype=np.uint32).reshape(len(Q), k).astype(np.int32)
        lp.unlink(missing_ok=True)
        rows.append({"param": p, "ms_per_query": w / len(Q) * 1000,
                     "qps": len(Q) / w, "inner": row["inner"], "I": I})
        print(f"    p={p:5d}  {w/len(Q)*1000:.4f} ms/q  [C++ · {N_THREADS} thread]", flush=True)
    qp.unlink(missing_ok=True)
    return rows


def probe_one(fam, ip, Qh, param, tag, reps=None):
    """One parameter, timed on the stratum's own queries. Returns (row, labels).

    Passes are combined by minimum, which buys precision where a pass is short
    enough for scheduling noise to matter. A deep point is not: at hundreds of
    milliseconds a query the harness's own inner loop already runs past its
    three-second floor once, and three outer passes buy nothing for three times
    the cost. Deep probes therefore take one pass, and the realised count is
    recorded per row.
    """
    if reps is None:
        reps = 1 if param >= DEEP_REPS_FROM else 3
    if fam == "vamana":
        return vamana_search(ip, Qh, param, tag, reps=reps)
    if fam == "hnswlib":
        r = timed_grid(ip, Qh, [param], k=10, reps=reps)[0]
        return ({"param": r["ef"], "ms_per_query": r["ms_per_query"],
                 "qps": r["qps"], "reps": reps}, r["labels"])
    r = faiss_bench(ip, Qh, [param], 10, tag, reps=reps)[0]
    return ({**{k: v for k, v in r.items() if k != "I"}, "reps": reps}, r["I"])


def attain(fam, ip, Qh, gt_h, rows, R, tag, pmax, tol=0.002, max_probes=14):
    """The measured cost of reaching recall R on this stratum -- no interpolation.

    Two phases, both measured. If nothing on the grid reaches R the parameter is
    doubled until something does or the parameter ceiling is hit; then the
    bracket is bisected until the cheapest point at or above R is within `tol`.
    Every probe is run and timed, so what is reported is a parameter that was
    executed, the recall it returned and the milliseconds it cost. Nothing is
    extrapolated, and nothing is abandoned for being slow -- only a target the
    index cannot reach at all is recorded as unreached.
    """
    def pts():
        return sorted({r["param"]: (r["ms_per_query"], r["recall"])
                       for r in rows}.items())

    def probe(param):
        row, I = probe_one(fam, ip, Qh, param, tag + f"_t{int(R * 100)}_{param}")
        row["recall"] = float(recall_at_k(I, gt_h, 10).mean())
        rows.append({**row, "probe_for": R})
        print(f"    probe param={param} -> {row['ms_per_query']:.3f} ms/q  "
              f"recall {row['recall']:.4f}", flush=True)
        return row

    probes = 0
    while not [1 for _, v in pts() if v[1] >= R]:
        deepest = pts()[-1][0]
        if deepest >= pmax or probes >= max_probes:
            p_, (ms, rc) = pts()[-1]
            return {"target": R, "reached": False, "deepest_param": p_,
                    "deepest_ms": ms, "deepest_recall": rc,
                    "stopped_by": "param_ceiling" if deepest >= pmax
                    else "probe_budget"}
        probe(min(deepest * 2, pmax)); probes += 1

    below = [(p_, v) for p_, v in pts() if v[1] < R]
    above = [(p_, v) for p_, v in pts() if v[1] >= R]
    lo = below[-1][0] if below else None
    hi, (hi_ms, hi_rc) = above[0]
    while lo is not None and hi - lo > 1 and hi_rc - R > tol and probes < max_probes:
        mid = (lo + hi) // 2
        row = probe(mid); probes += 1
        if row["recall"] >= R:
            hi, hi_ms, hi_rc = mid, row["ms_per_query"], row["recall"]
        else:
            lo = mid
    return {"target": R, "reached": True, "param": hi, "ms_per_query": hi_ms,
            "recall": hi_rc, "overshoot": hi_rc - R, "over_64ms": hi_ms > MS_NOTE,
            "bracket_below_param": lo, "probes": probes, "interpolated": False}


def faiss_degree(path, n, R):
    """faiss exposes graph adjacency only as an opaque pointer; read the measured
    degree back from the serialised index instead."""
    raw = np.fromfile(path, dtype=np.uint8)
    for K in range(R + 8, max(R - 2, 1), -1):
        for o in range(0, 512):
            nb = n * K * 4
            if o + nb > raw.nbytes:
                continue
            a = raw[o:o + nb].view(np.int32).reshape(n, K)
            if a.min() < -1 or a.max() >= n:
                continue
            d = (a >= 0).sum(1)
            if d.min() >= 1 and d.max() <= K:
                return float(d.mean())
    return None


def build_family(fam, corpus, arm, X, R=25, efc=200, M=16, nlist=4096):
    """Returns (index_path_or_None, build_seconds, params, measured_out_degree)."""
    WORK.mkdir(parents=True, exist_ok=True)
    tag = f"{fam}_{corpus}__{arm}"
    if fam == "hnswlib":
        ip = WORK / f"{tag}.bin"
        with timing_lock():
            _, bt, n, d = hnswlib_build(corpus, arm, M, efc, seed=100,
                                        index_path=ip, threads=BUILD_THREADS)
        from hnsw_io import level0_csr_by_label
        deg = float(np.diff(level0_csr_by_label(str(ip))[0]).mean())
        return ip, bt, {"M": M, "efConstruction": efc}, deg
    import faiss
    faiss.omp_set_num_threads(BUILD_THREADS)
    n, dim = X.shape
    ip = WORK / f"{tag}.faiss"
    with timing_lock():
        t0 = time.time()
        if fam == "faiss-hnsw":
            ix = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_L2)
            ix.hnsw.efConstruction = efc
            ix.add(X); par = {"M": M, "efConstruction": efc}
        elif fam == "faiss-nsg":
            ix = faiss.IndexNSGFlat(dim, R, faiss.METRIC_L2)
            ix.GK = 64
            ix.build_type = 1      # build_type 0 is an O(n^2) kNN graph
            ix.add(X); par = {"R": R, "GK": 64}
        elif fam == "faiss-ivf":
            quant = faiss.IndexFlatL2(dim)
            ix = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_L2)
            # k-means over all 1M points at 1024 dims does not finish in
            # reasonable time; faiss's own convention is to train on a sample.
            ix.cp.max_points_per_centroid = 64
            ix.cp.niter = 12
            ix.train(X); ix.add(X)
            par = {"nlist": nlist, "train_points_per_centroid": 64,
                   "kmeans_iters": 12}
        else:
            raise ValueError(fam)
        bt = time.time() - t0
    faiss.write_index(ix, str(ip))
    deg = None
    if fam in ("faiss-hnsw", "faiss-nsg"):
        deg = faiss_degree(ip, n, R if fam == "faiss-nsg" else 2 * M)
    del ix
    # shrink the build's thread pool before anything is timed beside it
    faiss.omp_set_num_threads(1)
    return ip, bt, par, deg


def build_vamana(X, tag, R=25, Lb=200, alpha=1.2):
    """Build the Vamana graph and keep it. Returns (prefix, build_s, params, deg)."""
    WORK.mkdir(parents=True, exist_ok=True)
    xp = WORK / f"{tag}_base.fbin"
    pref = WORK / f"{tag}_vamana"
    write_fbin(xp, X)
    t0 = time.time()
    with timing_lock():
        subprocess.run([str(DISKANN / "build_memory_index"),
                        "--data_type", "float", "--dist_fn", "l2",
                        "--data_path", str(xp),
                        "--index_path_prefix", str(pref), "-R", str(R),
                        "-L", str(Lb), "--alpha", str(alpha),
                        "-T", str(BUILD_THREADS)], check=True, capture_output=True)
    bt = time.time() - t0
    xp.unlink(missing_ok=True)
    raw = np.fromfile(pref, dtype=np.uint8)
    p, degs = 8 + 4 + 4 + 8, []
    while p + 4 <= raw.nbytes:
        dg = int(raw[p:p + 4].view(np.uint32)[0]); degs.append(dg); p += 4 + 4 * dg
    deg = float(np.mean(degs)) if degs else None
    return pref, bt, {"R": R, "L_build": Lb, "alpha": alpha}, deg


def _vamana_pass(pref, qp, nq, L, pre, tag):
    """One invocation of DiskANN's search binary. Returns (ms/query, labels)."""
    rp = WORK / f"{tag}_res_{L}"
    r = subprocess.run(pre + [str(DISKANN / "search_memory_index"),
                              "--data_type", "float", "--dist_fn", "l2",
                              "--index_path_prefix", str(pref),
                              "--query_file", str(qp), "-K", "10", "-L", str(L),
                              "--result_path", str(rp), "-T", str(N_THREADS)],
                       check=True, capture_output=True, text=True)
    qps = None
    for line in r.stdout.splitlines():
        m = re.match(r"\s*%d\s+([\d.]+)" % L, line)
        if m:
            qps = float(m.group(1))
    if qps is None:
        raise SystemExit(f"could not parse DiskANN's QPS for L={L}\n"
                         f"{r.stdout[-1500:]}")
    v = np.fromfile(str(rp) + f"_{L}_idx_uint32.bin", dtype=np.uint32)
    I = v[2:].reshape(-1, 10)[:nq].astype(np.int32)
    for f in WORK.glob(f"{tag}_res_{L}*"):
        f.unlink(missing_ok=True)
    return 1000.0 / qps, I


def vamana_search(pref, Q, L, tag, reps=3, floor_s=3.0, max_rows=200_000):
    """One search budget L, timed on this query set. Returns (row, labels).

    DiskANN times itself inside its own binary, so this path never went through
    timed_grid(). It takes the same machine-wide timing lock and the same pinned
    core the other families take -- a run measured while another lane was busy
    came back with L=96 slower than L=128, which cannot happen on a reserved
    core -- and it enforces the same three-second floor. The binary makes exactly
    one pass over the query file, so on a 100-query stratum at a small budget
    that pass lasts a few milliseconds and measures scheduling noise. The query
    set is therefore tiled until one pass runs past the floor, and the labels are
    read from the first copy. Passes are combined by minimum.
    """
    n = len(Q)
    qp = WORK / f"{tag}_q.fbin"
    write_fbin(qp, Q)
    pre = ["taskset", "-c", str(PIN_CORE)] if EXCLUSIVE else []
    qt = None
    try:
        with timing_lock():
            ms, I = _vamana_pass(pref, qp, n, L, pre, tag)
            tile = max(1, min(max_rows // n, int(floor_s * 1000.0 / (n * ms))))
            if tile > 1:
                qt = WORK / f"{tag}_qt.fbin"
                write_fbin(qt, np.tile(Q, (tile, 1)))
                ms = min(_vamana_pass(pref, qt, n, L, pre, tag)[0]
                         for _ in range(max(1, reps)))
            else:
                for _ in range(max(1, reps) - 1):
                    ms = min(ms, _vamana_pass(pref, qp, n, L, pre, tag)[0])
    finally:
        qp.unlink(missing_ok=True)
        if qt is not None:
            qt.unlink(missing_ok=True)
    print(f"    L={L:5d}  {ms:.4f} ms/q  [DiskANN, {N_THREADS} thread, "
          f"{n * tile:,} queries x min of {max(1, reps) if tile > 1 else reps}]",
          flush=True)
    return {"param": L, "ms_per_query": ms, "qps": 1000.0 / ms,
            "reps": max(1, reps), "timed_queries": int(n * tile)}, I


def vamana_grid(pref, Q, params, tag, keep_I=True):
    rows = []
    for L in params:
        if L < 10:
            continue
        row, I = vamana_search(pref, Q, L, tag,
                               reps=1 if L >= DEEP_REPS_FROM else 3)
        rows.append({**row, **({"I": I} if keep_I else {})})
    # A search budget that is bigger cannot be faster: if this trips, the timed
    # region shared the machine with something and the row is not a measurement.
    bad = [(a["param"], b["param"]) for a, b in zip(rows, rows[1:])
           if b["ms_per_query"] < a["ms_per_query"]]
    if bad:
        print(f"    !! timing not monotone in L at {bad} -- contended run", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True); ap.add_argument("--arm", required=True)
    ap.add_argument("--family", choices=list(FAMILIES), required=True)
    ap.add_argument("--stock-arm", default="vanilla")
    ap.add_argument("-R", type=int, default=25); ap.add_argument("-M", type=int, default=16)
    ap.add_argument("--nlist", type=int, default=4096)
    ap.add_argument("--time-hard", action="store_true",
                    help="Also time the search on the hard decile's own queries; "
                         "the stratum runs ~1.17x the mean ms/query, so its "
                         "Recall-SearchTime curve cannot borrow the full set's "
                         "x axis. See run_cell.py --time-hard.")
    ap.add_argument("--tag", default="",
                    help="Suffix on the output name. The own-decile runs carry "
                         "'__own' so they never overwrite the fixed-stratum "
                         "results the current draft is built on.")
    a = ap.parse_args()

    X = np.asarray(np.load(EMB / f"{a.corpus}__{a.arm}_docs.npy", mmap_mode="r"), dtype=np.float32)
    Q = np.load(EMB / f"{a.corpus}__{a.arm}_queries.npy").astype(np.float32)
    gt = np.load(RES / f"gt_{a.corpus}__{a.arm}_k10.npz")["idx"]
    gt_stock = np.load(RES / f"gt_{a.corpus}__{a.stock_arm}_k10.npz")["idx"]
    hard = np.load(RES / f"harddecile_{a.corpus}__{a.stock_arm}.npz")["hard"]
    rels = load_rels(a.corpus)
    rest = np.setdiff1d(np.arange(len(Q)), hard)
    tag = f"{a.family}_{a.corpus}__{a.arm}"
    hard_ms = {}; targets_hard = []; hard_extra = []
    print(f"[{a.family}/{a.corpus}/{a.arm}] n={len(X):,} build_threads={BUILD_THREADS} "
          f"search_threads={N_THREADS}", flush=True)

    # Vamana joins the shared path: its graph is built once and kept, and every
    # search budget goes through probe_one like every other family, so the deep
    # tail and the measured targets are the same measurement for all five.
    if a.family == "vamana":
        ip, bt, par, deg = build_vamana(X, tag, R=a.R)
    else:
        ip, bt, par, deg = build_family(a.family, a.corpus, a.arm, X,
                                        R=a.R, M=a.M, nlist=a.nlist)
    print(f"  built in {bt/60:.1f} min · out-degree "
          f"{('%.2f' % deg) if deg else 'n/a'}", flush=True)
    while (LAB / "work/GPU_LANE_BUSY").exists():
        time.sleep(20)
    if a.family == "vamana":
        rows = vamana_grid(ip, Q, L_GRID, tag)
    elif a.family == "hnswlib":
        g = timed_grid(ip, Q, EF_GRID, k=10, reps=3)
        rows = [{"param": r["ef"], "ms_per_query": r["ms_per_query"],
                 "qps": r["qps"], "I": r["labels"]} for r in g]
    else:
        params = NPROBE if a.family == "faiss-ivf" else EF_GRID
        _fast = [p_ for p_ in params if p_ < DEEP_REPS_FROM]
        _slow = [p_ for p_ in params if p_ >= DEEP_REPS_FROM]
        rows = []
        if _fast:
            rows += [{**r, "reps": 3} for r in
                     faiss_bench(ip, Q, _fast, 10, tag, reps=3)]
        if _slow:
            rows += [{**r, "reps": 1} for r in
                     faiss_bench(ip, Q, _slow, 10, tag, reps=1)]
    if a.time_hard:
        print(f"  timing the hard decile itself (n={len(hard)})", flush=True)
        if a.family == "vamana":
            hard_rows = vamana_grid(ip, Q[hard], L_GRID, tag + "_hard",
                                    keep_I=False)
        elif a.family == "hnswlib":
            hard_rows = [{"param": r["ef"], "ms_per_query": r["ms_per_query"],
                          "qps": r["qps"]}
                         for r in timed_grid(ip, Q[hard], EF_GRID, k=10, reps=3)]
        else:
            _p = NPROBE if a.family == "faiss-ivf" else EF_GRID
            _f = [x for x in _p if x < DEEP_REPS_FROM]
            _s = [x for x in _p if x >= DEEP_REPS_FROM]
            hard_rows = []
            for _lst, _rp in ((_f, 3), (_s, 1)):
                if not _lst:
                    continue
                hard_rows += [{**{k: v for k, v in r.items() if k != "I"},
                               "reps": _rp}
                              for r in faiss_bench(ip, Q[hard], _lst, 10,
                                                   tag + "_hard", reps=_rp)]
        hard_ms = {r["param"]: r for r in hard_rows}
        # The deep tail and the targets live on the decile, where the
        # comparison is read. The shallow grid left the baseline unmeasured
        # above its own ceiling, which is a gap in the measurement, not a
        # result -- so the parameter is pushed until the budget stops it.
        hr = []
        for r in rows:
            hm = hard_ms.get(r["param"])
            if hm:
                hr.append({**hm, "recall": float(
                    recall_at_k(r["I"], gt, 10)[hard].mean())})
        deep = []
        for p_ in DEEP:
            if p_ > (a.nlist if a.family == "faiss-ivf" else 10 ** 9):
                break
            row, I = probe_one(a.family, ip, Q[hard], p_, tag + "_deep")
            row["recall"] = float(recall_at_k(I, gt[hard], 10).mean())
            hr.append(row); deep.append(row)
        pmax = a.nlist if a.family == "faiss-ivf" else EF_MAX
        targets_hard = [attain(a.family, ip, Q[hard], gt[hard], hr, R,
                               tag, pmax) for R in TARGETS]
        for t in targets_hard:
            if t["reached"]:
                print(f"  recall {t['target']:.2f} measured at "
                      f"param={t['param']} -> {t['ms_per_query']:.3f} ms "
                      f"(recall {t['recall']:.4f}, {t['probes']} probes)",
                      flush=True)
            else:
                print(f"  recall {t['target']:.2f} UNREACHED; deepest "
                      f"{t['deepest_recall']:.4f} at "
                      f"{t['deepest_ms']:.1f} ms ({t['stopped_by']})", flush=True)
        hard_extra = sorted(hr, key=lambda r: r["param"])
    if a.family == "vamana":
        for f in WORK.glob(f"{tag}_vamana*"):
            f.unlink(missing_ok=True)
    else:
        ip.unlink(missing_ok=True)

    grid = []
    pq_rec, pq_param = [], []
    for r in rows:
        I = r.pop("I")
        rec = recall_at_k(I, gt, 10); mrr = mrr_at_k(I, rels, 10); nd = ndcg_at_k(I, rels, 10)
        gd = qrels_recall_at_k(I, rels, 10)
        rf = recall_at_k(I, gt_stock, 10)
        pq_rec.append(rec.astype(np.float32)); pq_param.append(r["param"])
        hm = hard_ms.get(r["param"])
        grid.append({**r,
                     **({"ms_per_query_hard": hm["ms_per_query"],
                         "qps_hard": hm["qps"]} if hm else {}),
                     "recall_all": float(rec.mean()), "recall_hard": float(rec[hard].mean()),
                     "recall_rest": float(rec[rest].mean()),
                     "recall_stock_targets_hard": float(rf[hard].mean()),
                     "mrr_all": float(mrr.mean()), "mrr_hard": float(mrr[hard].mean()),
                     "ndcg_all": float(nd.mean()), "ndcg_hard": float(nd[hard].mean()),
                     "qrels_recall_all": float(gd.mean()),
                     "qrels_recall_hard": float(gd[hard].mean())})
    out = {"corpus": a.corpus, "arm": a.arm, "family": a.family,
           "family_label": FAMILIES[a.family][0], "structure": FAMILIES[a.family][1],
           "params": par, "build_time_s": bt, "build_threads": BUILD_THREADS,
           "search_threads": N_THREADS, "measured_out_degree": deg,
           "timing_valid": EXCLUSIVE, "x_axis": "search budget (ef/nprobe)",
           "n_docs": int(len(X)), "n_hard": int(len(hard)),
           "hard_decile_arm": a.stock_arm, "timed_hard_decile": bool(hard_ms),
           "ms_cap": None, "targets": TARGETS,
           "grid": grid, "grid_hard": hard_extra, "targets_hard": targets_hard}
    nm = f"family_{a.family}_{a.corpus}__{a.arm}{a.tag}"
    (RES / f"{nm}.json").write_text(json.dumps(out, indent=2))
    # per-query recall, so a stratum can be re-cut without re-timing the index
    np.savez_compressed(RES / f"pq_{nm}.npz", param=np.array(pq_param),
                        recall=np.stack(pq_rec), hard=hard)
    print(f"wrote {nm}.json · build {bt/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
