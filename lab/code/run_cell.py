"""One measurement cell = (corpus, arm, M). Builds, times, and measures.

Emits results/cell_<corpus>__<arm>__M<M>.json with, per ef on the grid: measured
ms/query and q/s at a FIXED thread count, recall@10 against the exact top-10 of
the embedding this arm was measured with, MRR@10 and nDCG@10 against qrels --
each on the full query set and on the corpus's fixed hard decile.
"""
import argparse, gc, json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import EMB, ROOT  # noqa: E402
# LAB comes from paths.ROOT: SYN_ROOT if set, else this file's parent.
# It used to be a fixed absolute path, which resolved only on the machine
# this was written on, where that path was a symlink into the tree.
LAB = ROOT; RES = LAB / "results"
IDXDIR = Path(os.environ.get("SYN_INDEX_DIR", "/dev/shm/synidx"))
from metrics import recall_at_k, mrr_at_k, ndcg_at_k, qrels_recall_at_k          # noqa: E402
from index_hnsw import (build, timed_grid, EF_GRID, N_THREADS,        # noqa: E402
                        BUILD_THREADS, BUILD_THREADS_FULL, EXCLUSIVE)
from hnsw_io import level0_csr_by_label                       # noqa: E402
from kappa import phi                                          # noqa: E402


def load_rels(corpus):
    qs = [json.loads(l) for l in
          (LAB / "corpora" / f"{corpus}_queries.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return [{int(k): int(v) for k, v in q["rel"].items()} for q in qs]


def hard_decile(corpus, ref_arm="vanilla", M=16, efc=200, seed=777, frac=0.10,
                ef_sel=(32, 64, 128, 256, 512)):
    """The fixed difficulty stratum: lowest mean ANN recall against the stock
    encoder's exact top-10, on an index built independently of every cell that is
    later measured (distinct seed, never reused). Defined once per collection and
    held fixed across arms."""
    out = RES / f"harddecile_{corpus}__{ref_arm}.npz"
    if out.exists():
        z = np.load(out); return z["hard"], z["mean_recall"]
    IDXDIR.mkdir(parents=True, exist_ok=True)
    p = IDXDIR / f"sel_{corpus}.bin"
    print(f"[{corpus}] selector index (independent build, seed={seed})", flush=True)
    # not a reported number -- use the whole machine
    ix, bt, n, d = build(corpus, ref_arm, M, efc, seed=seed, index_path=None,
                         threads=BUILD_THREADS_FULL)
    Q = np.load(EMB / f"{corpus}__{ref_arm}_queries.npy").astype(np.float32)
    gt = np.load(RES / f"gt_{corpus}__{ref_arm}_k10.npz")["idx"]
    acc = np.zeros(len(Q))
    for ef in ef_sel:
        ix.set_ef(ef)
        lab, _ = ix.knn_query(Q, k=10, num_threads=BUILD_THREADS_FULL)
        acc += recall_at_k(lab.astype(np.int32), gt, 10)
    acc /= len(ef_sel)
    nh = int(round(frac * len(Q)))
    hard = np.sort(np.argsort(acc, kind="stable")[:nh]).astype(np.int32)
    np.savez_compressed(out, hard=hard, mean_recall=acc, seed=seed, M=M, efc=efc)
    print(f"[{corpus}] hard decile n={nh} mean recall {acc[hard].mean():.4f} "
          f"vs rest {acc[np.setdiff1d(np.arange(len(Q)), hard)].mean():.4f}", flush=True)
    del ix; gc.collect()
    return hard, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True); ap.add_argument("--arm", required=True)
    ap.add_argument("-M", type=int, required=True)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--reps", type=int, default=3,
                    help="Timed repetitions. One clean run on a quiet dedicated "
                         "machine is the ANN-benchmark convention; the harness "
                         "already repeats each sweep internally until the timed "
                         "region exceeds three seconds. Passes are combined by "
                         "MINIMUM, not mean: contention can only slow a pass "
                         "down, so the fastest is the cleanest estimate.")
    ap.add_argument("--phi-sample", type=int, default=0,
                    help="Size of the selection-free query sample phi is also "
                         "measured on. 0 = skip. The hard decile is always measured "
                         "in full and is what every table reports; the sample costs "
                         "~300 s a cell at M=32 and no table reads it.")
    ap.add_argument("--ef-extra", default="",
                    help="Extra ef points appended to the grid, comma separated. The "
                         "standard grid stops at 2048; a target the baseline does not "
                         "reach inside it is not unreachable, it is unmeasured, and "
                         "saying which needs the deeper points.")
    ap.add_argument("--tune", action="store_true",
                    help="Tuning mode: this cell picks a hyperparameter, it does not "
                         "go in a table. Build on every core, search on many threads "
                         "with one pass, and skip phi, the graph dump and the "
                         "per-query arrays. Recall and MRR are exact as ever; the "
                         "milliseconds are NOT a measurement and are marked as such "
                         "in the JSON, so nothing downstream can quote them.")
    ap.add_argument("--tune-threads", type=int,
                    default=int(os.environ.get("SYN_TUNE_THREADS", "32")),
                    help="Search threads in tuning mode.")
    ap.add_argument("--phi-hops", default="1,2,3,4",
                    help="Hop counts to measure fragmentation at. Reporting the "
                         "whole progression beats picking one: it is one BFS pass "
                         "per hop and it shows where the relation saturates.")
    ap.add_argument("--phi-stock-targets", action="store_true",
                    help="Also measure phi over the stock arm's target set. Off: "
                         "every arm is reported on the target set of the embedding "
                         "it was built from, so this third BFS pass feeds nothing.")
    ap.add_argument("--ef-grid", default="",
                    help="Replace the base ef grid, comma separated. The stock "
                         "grid doubles and is too coarse where the curves "
                         "actually separate: on the hard decile almost all of "
                         "the gap between arms sits below ef=512, and three "
                         "points across 128-512 draw that stretch as a straight "
                         "line it is not. The deep tail (768+) is still appended, "
                         "as are --ef-extra and --ef-deep.")
    ap.add_argument("--ef-deep", default="",
                    help="Deep ef points timed on the HARD DECILE ONLY, comma "
                         "separated. The two strata saturate at very different "
                         "budgets: on MS MARCO M=16 all queries are at 0.992 by "
                         "ef=3072 while the decile needs ef=12288 to clear 0.99. "
                         "Timing those depths on the full query set costs ~10x "
                         "what it costs on the decile and buys a flat line, so "
                         "the deep tail is measured where it still moves. These "
                         "points extend the decile curve only and are reported "
                         "separately, never merged into the full-set grid.")
    ap.add_argument("--ms-cap", type=float, default=0.0,
                    help="Latency budget, ms/query on one core. Deep points are "
                         "measured in increasing ef and the tail stops as soon "
                         "as a point costs more than this; anything past it is "
                         "reported as out of range rather than measured. A "
                         "single-threaded ANN query that costs more than a few "
                         "tens of milliseconds has left the regime the index is "
                         "for, so spending hours pinning down its exact recall "
                         "buys a number no deployment would use. 0 disables.")
    ap.add_argument("--time-hard", action="store_true",
                    help="Also time the search on the hard decile's own queries. "
                         "The stratum is NOT as cheap as the full set: measured "
                         "back-to-back on one FEVER M=16 index the decile costs "
                         "1.15-1.19x the mean ms/query at every ef, because a "
                         "query that lands badly traverses more of the graph "
                         "before the heap settles. A Recall-SearchTime curve "
                         "drawn for the stratum therefore cannot borrow the full "
                         "set's x axis -- it would flatter the stratum by ~17%.")
    ap.add_argument("--sat", type=float, default=0.999,
                    help="Stop the deep tail once the stratum's recall reaches "
                         "this. A saturated curve has nothing left to measure; "
                         "1.0 disables and runs every requested point.")
    ap.add_argument("--tag", default="")
    ap.add_argument("-k", type=int, default=10,
                    help="Retrieval depth. k=10 is the paper's setting; k=100 is "
                         "the candidate-generation depth a reranking pipeline "
                         "actually uses, and recall@100 is the ceiling no "
                         "reranker can exceed. The difficulty stratum stays "
                         "defined at k=10 so the two are comparable.")
    ap.add_argument("--k-eval", default="",
                    help="Extra depths to score the SAME returned list at, comma "
                         "separated. In HNSW ef is the search budget and k only "
                         "truncates the result heap, so one search at the largest "
                         "depth gives every shallower depth by prefix -- no "
                         "rebuild, and every depth is compared at an identical "
                         "traversal.")
    ap.add_argument("--phi-hard-only", action="store_true",
                    help="phi over a 100-document target set is 10x the BFS work; "
                         "restrict it to the fixed hard decile.")
    ap.add_argument("--stock-arm", default="vanilla",
                    help="Baseline arm: defines the hard decile and the fixed "
                         "target set. Per encoder when the encoder changes.")
    a = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True); IDXDIR.mkdir(parents=True, exist_ok=True)
    name = f"{a.corpus}__{a.arm}__M{a.M}{a.tag}" + (f"__k{a.k}" if a.k != 10 else "")
    t_all = time.time()
    hard, sel_recall = hard_decile(a.corpus, ref_arm=a.stock_arm)
    rels = load_rels(a.corpus)
    gt = np.load(RES / f"gt_{a.corpus}__{a.arm}_k{a.k}.npz")["idx"]
    gt_stock = np.load(RES / f"gt_{a.corpus}__{a.stock_arm}_k{a.k}.npz")["idx"]
    Q = np.load(EMB / f"{a.corpus}__{a.arm}_queries.npy").astype(np.float32)
    nq = len(Q)
    rest = np.setdiff1d(np.arange(nq), hard)

    ipath = IDXDIR / f"{name}.bin"
    side = IDXDIR / f"{name}.build.json"
    if ipath.exists() and side.exists():
        # A prebuild lane made this while the queue was busy elsewhere. Building
        # is not the measured quantity, so it does not have to happen here.
        meta = json.loads(side.read_text())
        bt, n, d = meta["build_time_s"], meta["n_docs"], meta["dim"]
        print(f"[{name}] reusing prebuilt index ({bt:.0f}s build, prebuild lane)", flush=True)
    else:
        print(f"[{name}] build M={a.M} efC={a.efc} build_threads={BUILD_THREADS} "
              f"(pinned: build time is reported) search_threads={N_THREADS}", flush=True)
        ix, bt, n, d = build(a.corpus, a.arm, a.M, a.efc, seed=a.seed, index_path=ipath,
                             threads=BUILD_THREADS_FULL if a.tune else None)
        print(f"[{name}] built in {bt:.0f}s", flush=True)
        del ix

    gc.collect()                  # the native harness owns the index while timing
    # Only the TIMED section needs the machine to itself. Building alongside the
    # GPU prep lane is fine and saves an hour of otherwise idle accelerator.
    while (LAB / "work/GPU_LANE_BUSY").exists():
        print("  GPU lane busy, holding before timing...", flush=True); time.sleep(20)
    # ef must be at least k; for a deep candidate list the standard grid runs out,
    # so it is extended rather than truncated to three points
    extra = [int(x) for x in a.ef_extra.split(",") if x.strip()]
    base = [int(x) for x in a.ef_grid.split(",") if x.strip()] or EF_GRID
    ef_grid = sorted({e for e in base + [640, 768, 896, 1024, 1280, 1536, 2048] + extra
                      if e >= a.k}) or [a.k]
    grid = timed_grid(ipath, Q, ef_grid, k=a.k,
                      reps=1 if a.tune else a.reps,
                      threads=a.tune_threads if a.tune else None,
                      pin=not a.tune)
    # The stratum's own x axis. Same index, same pinned core, same lock, one
    # extra pass over 10% of the queries -- so it costs a tenth of the timing
    # and removes the only quantity the hard-decile curve was borrowing.
    hard_ms = {}
    if a.time_hard and not a.tune:
        print(f"[{name}] timing the hard decile itself (n={len(hard)})", flush=True)
        # The decile curve takes its y from the full-set pass (all queries
        # searched, then cut to the decile) and its x from this pass. That is
        # only sound if the two passes return the same neighbours -- HNSW search
        # should be deterministic in (index, query, ef) with one thread and no
        # randomness, but "should be" is an assumption the curve rests on. This
        # pass already has the labels, so checking costs a recall computation
        # and nothing else. Recorded per cell rather than asserted once.
        gt_h = gt[hard]
        for g in timed_grid(ipath, Q[hard], ef_grid, k=a.k, reps=a.reps, pin=True):
            lab_h = g.pop("labels", None)
            if lab_h is not None:
                g["recall_hard_selfpass"] = float(
                    recall_at_k(lab_h, gt_h, a.k).mean())
            hard_ms[g["ef"]] = g

    rows = []
    # Per-query recall/MRR at every ef, kept so the distribution over queries can
    # be drawn without rebuilding the index. 11 ef x nq floats is a few hundred kB.
    pq_rec, pq_mrr, pq_ef = [], [], []
    for g in grid:
        lab = g.pop("labels")
        rec = recall_at_k(lab, gt, a.k)
        mrr = mrr_at_k(lab, rels, 10)          # quality is always MRR@10
        ndcg = ndcg_at_k(lab, rels, 10)
        rec_fixed = recall_at_k(lab, gt_stock, a.k)
        # what a reranker is handed: relevant documents present in the returned list
        qrec = qrels_recall_at_k(lab, rels, a.k)
        depths = {}
        for dd in sorted({int(x) for x in a.k_eval.split(",") if x.strip()}):
            if dd > a.k:
                continue
            q = qrels_recall_at_k(lab, rels, dd)
            r = recall_at_k(lab, gt[:, :dd], dd)
            depths[str(dd)] = {"qrels_recall": float(q.mean()),
                               "qrels_recall_hard": float(q[hard].mean()),
                               "recall_hard": float(r[hard].mean()),
                               "recall_all": float(r.mean())}
        pq_rec.append(rec.astype(np.float32)); pq_mrr.append(mrr.astype(np.float32))
        pq_ef.append(g["ef"])
        hm = hard_ms.get(g["ef"])
        rows.append({**{k: v for k, v in g.items()},
                     **({"ms_per_query_hard": hm["ms_per_query"],
                         "qps_hard": hm["qps"],
                         "wall_s_hard": hm["wall_s"],
                         "recall_hard_selfpass": hm.get("recall_hard_selfpass")}
                        if hm else {}),
                     "recall_all": float(rec.mean()), "recall_hard": float(rec[hard].mean()),
                     "recall_rest": float(rec[rest].mean()),
                     "recall_stock_targets_all": float(rec_fixed.mean()),
                     "recall_stock_targets_hard": float(rec_fixed[hard].mean()),
                     # which arm defined the hard decile and the fixed
                     # targets. Not recording it cost an afternoon of
                     # checking whether the encoder axis compared each
                     # encoder against its own decile or against bge-m3's.
                     "stock_arm": a.stock_arm,
                     "by_depth": depths,
                     "qrels_recall_all": float(qrec.mean()),
                     "qrels_recall_hard": float(qrec[hard].mean()),
                     "mrr_all": float(mrr.mean()), "mrr_hard": float(mrr[hard].mean()),
                     "mrr_rest": float(mrr[rest].mean()),
                     "ndcg_all": float(ndcg.mean()), "ndcg_hard": float(ndcg[hard].mean())})

    # The deep tail, on the decile's own queries. Same index, same pinned core,
    # same lock -- only the query set and the depth differ.
    deep_rows = []
    for e in sorted({int(x) for x in a.ef_deep.split(",") if x.strip()}):
        if e <= max(ef_grid):
            continue
        deep_rows.append(e)
    if deep_rows and not a.tune:
        print(f"[{name}] deep tail on the hard decile (n={len(hard)}): "
              f"{deep_rows}", flush=True)
        gt_h = gt[hard]
        rels_h = [rels[i] for i in hard]
        deep = []
        # One ef at a time, so the budget can stop the tail. ms/query is
        # monotone in ef, so the first point over the cap is also the last one
        # worth paying for -- every deeper point costs more and is further
        # outside the regime.
        for e in deep_rows:
            g = timed_grid(ipath, Q[hard], [e], k=a.k, reps=a.reps, pin=True)[0]
            lab = g.pop("labels")
            deep.append({"ef": g["ef"], "ms_per_query_hard": g["ms_per_query"],
                         "qps_hard": g["qps"], "wall_s_hard": g["wall_s"],
                         "recall_hard": float(recall_at_k(lab, gt_h, a.k).mean()),
                         "mrr_hard": float(mrr_at_k(lab, rels_h, 10).mean()),
                         "qrels_recall_hard": float(
                             qrels_recall_at_k(lab, rels_h, a.k).mean())})
            if a.ms_cap and g["ms_per_query"] > a.ms_cap:
                print(f"[{name}] ef={e} costs {g['ms_per_query']:.1f} ms/q > "
                      f"{a.ms_cap:g} ms cap -- stopping the tail; deeper points "
                      f"are out of range", flush=True)
                break
            # Saturation, not latency: once the stratum returns essentially all
            # of its own top-k there is nothing left for a deeper budget to
            # find, and every further point costs more for a flat line. This
            # stops on the measurement, not on a budget -- the curve is finished.
            if deep[-1]["recall_hard"] >= a.sat:
                print(f"[{name}] ef={e} reaches recall {deep[-1]['recall_hard']:.4f} "
                      f">= {a.sat:g} -- tail saturated, stopping", flush=True)
                break
        deep_rows = deep
    else:
        deep_rows = []

    ipd, idd, h = level0_csr_by_label(str(ipath))
    outdeg = float(np.diff(ipd).mean())
    # In-degree is where a rearrangement of neighbourhoods shows up: out-degree is
    # capped by the builder and barely moves, but who gets pointed AT is free to
    # concentrate or spread. Kept as summary statistics, not the whole histogram.
    indeg = np.bincount(idd, minlength=len(ipd) - 1).astype(np.float64)
    indeg_stats = {"mean": float(indeg.mean()), "sd": float(indeg.std()),
                   "median": float(np.median(indeg)),
                   "p90": float(np.quantile(indeg, 0.90)),
                   "p99": float(np.quantile(indeg, 0.99)),
                   "max": float(indeg.max()),
                   "share_zero": float((indeg == 0).mean()),
                   "gini": float((np.abs(np.subtract.outer(
                       np.random.default_rng(7).choice(indeg, 4000),
                       np.random.default_rng(8).choice(indeg, 4000))).mean())
                       / (2 * max(indeg.mean(), 1e-9)))}
    print(f"[{name}] measured out-degree {outdeg:.2f} (maxM0={h['maxM0']})", flush=True)

    # The index on disk, in bytes: an arm that repairs connectivity by adding edges
    # pays here, and an arm that moves vectors does not. This column is where that
    # difference is settled rather than argued.
    index_bytes = ipath.stat().st_size if ipath.exists() else None
    # The exhaustive ceiling: what this embedding could deliver with no index at
    # all. Arms that change the ranking (any post-hoc transform) move it, and the
    # graph number alone would then mislead.
    exh = {}
    try:
        gt_idx = gt[:, :a.k]
        exh_mrr = mrr_at_k(gt_idx.astype(np.int32), rels, 10)
        exh_ndcg = ndcg_at_k(gt_idx.astype(np.int32), rels, 10)
        exh_qrec = qrels_recall_at_k(gt_idx.astype(np.int32), rels, a.k)
        exh = {"mrr_all": float(exh_mrr.mean()), "mrr_hard": float(exh_mrr[hard].mean()),
               "ndcg_all": float(exh_ndcg.mean()),
               "qrels_recall_all": float(exh_qrec.mean()),
               "qrels_recall_hard": float(exh_qrec[hard].mean())}
    except Exception as e:
        print(f"[{name}] exhaustive ceiling skipped: {e}", flush=True)

    rec = {"corpus": a.corpus, "arm": a.arm, "M": a.M, "efConstruction": a.efc,
           "index_bytes": index_bytes, "exhaustive": exh,
           "build_seed": a.seed, "n_docs": int(n), "dim": int(d), "n_queries": int(nq),
           "search_threads": N_THREADS, "build_threads": BUILD_THREADS,
           "reps": 1 if a.tune else a.reps, "build_time_s": bt,
           "tuning": bool(a.tune),
           "timing_valid": (not a.tune) and EXCLUSIVE,
           "x_axis": "search budget (ef)",
           "build_threads_used": BUILD_THREADS_FULL if a.tune else BUILD_THREADS,
           "timing_harness": "cpp/hnsw_bench (native, persistent thread pool)",
           "measured_out_degree": outdeg, "maxM0": int(h["maxM0"]),
           "in_degree": indeg_stats,
           "n_hard": int(len(hard)), "k": a.k,
           "hard_decile_arm": a.stock_arm, "ms_cap": a.ms_cap or None,
           "decile_pass_max_recall_delta": (
               max((abs(r["recall_hard"] - r["recall_hard_selfpass"])
                    for r in rows if r.get("recall_hard_selfpass") is not None),
                   default=None)),
           "timed_hard_decile": bool(hard_ms),
           "prebuilt": bool(side.exists()), "grid": rows,
           "grid_deep_hard": deep_rows}

    if not a.tune:
        rng = np.random.default_rng(20260809)
        # A 3-hop BFS visits ~deg^3 nodes per seed, so at M=32 on 8.8M documents
        # one seed touches a quarter-million nodes. The hard decile is always
        # measured in full; the selection-free sample is what gets trimmed.
        samp = (np.sort(rng.choice(nq, min(a.phi_sample, nq), replace=False))
                if a.phi_sample > 0 else np.zeros(0, dtype=np.int64))
        pops = {"hard": hard}
        if a.phi_sample > 0 and not a.phi_hard_only:
            pops["sample"] = samp
        rec["phi"] = {}
        for pname, pop in pops.items():
            t0 = time.time()
            # Climb the hop counts until the relation saturates. phi_h falls
            # monotonically in h and once it is at 1 the extra hop says nothing
            # -- and it is not cheap: at M=32 on 1M documents h=4 takes 664 s
            # against 42 s for h=3, to return 1.077. Stop at the first hop that
            # is essentially one piece.
            hops = sorted({int(x) for x in a.phi_hops.split(",") if x.strip()})
            P = {}
            for h in hops:
                P[h] = phi(ipd, idd, gt[pop], h=h, log_every=0)
                if float(P[h].mean()) < 1.05 and h < max(hops):
                    print(f"[{name}] phi/{pname}: saturated at h={h} "
                          f"({P[h].mean():.3f}); skipping deeper hops", flush=True)
                    break
            p3 = P.get(3, P[max(P)]); p1 = P.get(1, P[min(P)])
            p3f = (phi(ipd, idd, gt_stock[pop], h=3, log_every=0)
                   if a.phi_stock_targets else None)
            # phi is reported on THIS arm's exact top-k, measured on the index
            # built from this arm's embedding. phi3_stock_targets keeps the same
            # quantity over the stock arm's target set; it is stored, not reported.
            rec["phi"][pname] = {"n": int(len(pop))}
            for h, v in P.items():
                rec["phi"][pname][f"phi{h}"] = float(v.mean())
                rec["phi"][pname][f"pr_phi{h}_eq_1"] = float((v == 1).mean())
            if p3f is not None:
                rec["phi"][pname]["phi3_stock_targets"] = float(p3f.mean())
            # phi at every evaluated depth, on the SAME index: the target set is
            # nested (top-1 inside top-10 inside top-100), so this is the depth
            # axis of fragmentation, not a different measurement. Cheap: the BFS
            # cost scales with the number of seeds, and depth 1 is trivially 1.
            for dd in sorted({int(x) for x in a.k_eval.split(",") if x.strip()}):
                if dd > a.k:
                    continue
                d3 = phi(ipd, idd, gt[pop][:, :dd], h=3, log_every=0)
                d1 = phi(ipd, idd, gt[pop][:, :dd], h=1, log_every=0)
                rec["phi"][pname].setdefault("by_depth", {})[str(dd)] = {
                    "phi3": float(d3.mean()), "phi1": float(d1.mean()),
                    "pr_phi3_eq_1": float((d3 == 1).mean()),
                    "pr_phi1_eq_1": float((d1 == 1).mean())}
            print(f"[{name}] phi/{pname}: "
                  + "  ".join(f"phi{h}={v.mean():.3f}" for h, v in sorted(P.items()))
                  + f"  ({time.time()-t0:.0f}s)", flush=True)
        # the same phi on the queries the arm was TRAINED on, when their exact
        # top-k under this arm has been computed. Reported beside the held-out
        # number, never instead of it.
        gtr = RES / f"gttrain_{a.corpus}__{a.arm}_k{a.k}.npz"
        if gtr.exists():
            g_tr = np.load(gtr)["idx"]
            t0 = time.time()
            p3 = phi(ipd, idd, g_tr, h=3, log_every=0)
            p1 = phi(ipd, idd, g_tr, h=1, log_every=0)
            rec["phi"]["train_queries"] = {
                "phi3": float(p3.mean()), "phi1": float(p1.mean()),
                "pr_phi1_eq_1": float((p1 == 1).mean()),
                "pr_phi3_eq_1": float((p3 == 1).mean()), "n": int(len(g_tr))}
            print(f"[{name}] phi/train_queries: phi3={p3.mean():.4f} "
                  f"phi1={p1.mean():.4f} ({time.time()-t0:.0f}s)", flush=True)
        np.savez_compressed(RES / f"pops_{name}.npz", hard=hard, sample=samp)
    if a.tune:
        (RES / f"cell_{name}.json").write_text(json.dumps(rec, indent=2))
        ipath.unlink(missing_ok=True)
        (IDXDIR / f"{name}.build.json").unlink(missing_ok=True)
        g0 = sorted(rec["grid"], key=lambda r: r["ef"])[0]
        print(f"[{name}] TUNE done in {time.time()-t_all:.0f}s · "
              f"recall_hard(ef16) {g0['recall_hard']:.4f} · mrr_hard {g0['mrr_hard']:.4f}",
              flush=True)
        return
    np.savez_compressed(RES / f"perquery_{name}.npz",
                        ef=np.array(pq_ef), recall=np.stack(pq_rec),
                        mrr=np.stack(pq_mrr), hard=hard)

    # phi is already in the JSON; the raw adjacency is kept only for the arms that
    # get re-analysed -- Vanilla and the headline SYN arms. An ablation or sweep arm
    # exists for one number, and its dump is 40 s of compression and 100 MB of disk
    # for a file nothing opens.
    # Vanilla, the transfer arm, and whichever arm each collection is reported
    # with: those are the graphs that get re-analysed (phi at other hop counts,
    # in-degree, the mediation scatter). A sweep arm exists for one number and
    # its 100 MB dump would never be opened.
    _pa = RES / "paper_arms.json"
    _keep = {"vanilla", "syn", "syn-tr"}
    if _pa.exists():
        _keep |= {v["arm"] for v in json.loads(_pa.read_text()).values()}
    if n <= 1_200_000 and a.arm in _keep:
        np.savez_compressed(RES / f"graph_{name}.npz", indptr=ipd, indices=idd)
    (RES / f"cell_{name}.json").write_text(json.dumps(rec, indent=2))
    ipath.unlink(missing_ok=True)
    (IDXDIR / f"{name}.build.json").unlink(missing_ok=True)
    print(f"[{name}] DONE in {time.time()-t_all:.0f}s -> results/cell_{name}.json", flush=True)


if __name__ == "__main__":
    main()
