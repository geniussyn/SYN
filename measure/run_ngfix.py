"""HNSW-NGFix* on both embeddings: the baseline alone, and composed with SYN.

NGFix (SIGMOD'26) repairs the graph; SYN repairs the embedding the graph is built
from. They are not alternatives, so the interesting measurement is the 2x2: each
fix alone, neither, and both. Four indexes per arm come out of one bottom graph:

    bottom   the paper's own HNSW, M=16 MEX=48 efC=500, no repair
    ngfix    Neighbouring Graph Defects Fixing only
    rfix     Reachability Fixing only
    both     HNSW-NGFix*, the paper's method

The stage split is ours (test/build_hnsw_ngfix_stage.cc, one flag on their own
build driver); everything else is their code at their defaults, including the
efC_AKNN=1500 approximate ground truth the build derives for itself.

MILLISECONDS HERE ARE THEIR HARNESS, NOT OURS. search_hnsw_ngfix times each query
with gettimeofday and makes one pass; our figures come from cpp/hnsw_bench, which
repeats until three seconds have elapsed. The two are not interchangeable, so
this file's numbers are only ever compared with each other -- which is enough,
because every arm in the 2x2 goes through the same binary. The pinned core, the
machine-wide lock and the min-of-three are ours, applied from outside.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from index_hnsw import EXCLUSIVE, PIN_CORE, timing_lock         # noqa: E402
from paths import ROOT                                          # noqa: E402

BIN = Path("/root/baselines/ngfix/build/test")
DATA = Path("/dev/shm/ngfix")
IDX = DATA / "idx"
RES = ROOT / "results"
# The NGFix scripts' own defaults. efC is overridable from the command line
# because it is a BUILD HYPERPARAMETER, not part of their method: their driver
# ships 500 and every other arm in the paper is built at 200, and comparing
# across that is comparing two builders rather than two interventions. MEX is
# left alone -- it is the cap on how many repair edges NGFix may add to a node
# (node.h:201 prunes at ngfix_sz == MEX), so it IS the method, and the memory
# it costs is measured below rather than argued about.
M, MEX, EFC, EFC_AKNN = 16, 48, 500, 1500
STAGES = ["bottom", "ngfix", "rfix", "both"]


def run(cmd, lock=True, pin=False):
    pre = ["taskset", "-c", str(PIN_CORE)] if (pin and EXCLUSIVE) else []
    with timing_lock(lock):
        t0 = time.time()
        r = subprocess.run(pre + [str(c) for c in cmd], capture_output=True,
                           text=True)
    if r.returncode != 0:
        raise SystemExit(f"{cmd[0]} failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return r, time.time() - t0


def build_all(tag, metric, itag=None, efc=None):
    """One bottom graph, then one repaired copy per stage.

    Returns (build seconds, index bytes), both per stage. The bytes are the
    point of recording: `bottom` is an HNSW at M=16 and the three repaired
    copies are that graph plus up to MEX extra edges per node, so the purple arm
    does not sit at the same index size as the rest of the figure and the
    difference has to be reported with the curve rather than assumed away.
    """
    IDX.mkdir(parents=True, exist_ok=True)
    efc = EFC if efc is None else efc
    bt = {}
    itag = itag or tag
    bottom = IDX / f"{tag}_bottom.index"      # history-independent, shared
    if not bottom.exists():
        print(f"  [{tag}] bottom graph M={M} MEX={MEX} efC={efc}", flush=True)
        _, s = run([BIN / "build_hnsw_bottom",
                    "--base_data_path", DATA / f"{tag}_base.fbin",
                    "--metric", metric, "--result_hnsw_index_path", bottom,
                    "--M", M, "--MEX", MEX, "--efC", efc])
        bt["bottom"] = s
    for st in ("ngfix", "rfix", "both"):
        p = IDX / f"{itag}_{st}.index"
        if p.exists():
            continue
        print(f"  [{tag}] {st} from the bottom graph, efC_AKNN={EFC_AKNN}", flush=True)
        _, s = run([BIN / "build_hnsw_ngfix_stage",
                    "--train_query_path", DATA / f"{tag}_hist.fbin",
                    "--base_graph_path", bottom, "--metric", metric,
                    "--efC_AKNN", EFC_AKNN, "--stage", st,
                    "--result_index_path", p])
        bt[st] = s
    nb = {"bottom": bottom.stat().st_size if bottom.exists() else None}
    for st in ("ngfix", "rfix", "both"):
        q = IDX / f"{itag}_{st}.index"
        nb[st] = q.stat().st_size if q.exists() else None
    if nb["bottom"]:
        print("  [%s] index bytes: %s" % (tag, "  ".join(
            f"{k} {v / 2**20:.0f} MiB ({v / nb['bottom']:.2f}x)"
            for k, v in nb.items() if v)), flush=True)
    return bt, nb


# The efs grid their k=10 driver hard-codes; passed explicitly to the timed
# binary, which takes it as a flag rather than compiling it in.
EFS_K10 = "10,15,20,30,40,50,60,70,80,90,100,120,150,180,200,300,400,500,800,1000"


def search(tag, stage, pop, k=10, reps=3, itag=None, timed=False):
    """The efs grid on one stratum, min of `reps` passes. Returns rows.

    `timed=True` swaps their search driver for search_hnsw_ngfix_timed, which is
    their search call under cpp/hnsw_bench's clock: a warm-up sweep, a
    calibration pass, enough inner repetitions to fill three seconds, and the
    minimum of three such regions, with the sweep timed as a whole instead of
    each query wrapped in its own gettimeofday pair. Recall and ndc still come
    from their evaluation path. Only with this are the milliseconds on the same
    axis as every other arm in the paper; their own driver's are not, and say so
    in this file's header.

    The binary does its own repetitions and keeps its own minimum, so the outer
    loop runs once -- repeating it would be a minimum of minima and would quietly
    buy a faster number than the protocol allows.
    """
    idx = IDX / f"{(itag or tag) if stage != 'bottom' else tag}_{stage}.index"
    out = DATA / f"{tag}_{stage}_{pop}.csv"
    best = {}
    binary = "search_hnsw_ngfix_timed" if timed else "search_hnsw_ngfix"
    extra = ["--efs", EFS_K10, "--min_sec", 3.0, "--reps", 3] if timed else []
    for _ in range(1 if timed else reps):
        r, _ = run([BIN / binary,
                    "--test_query_path", DATA / f"{tag}_{pop}.fbin",
                    "--test_gt_path", DATA / f"{tag}_{pop}_gt.ibin",
                    "--metric", "ip_float", "--K", k,
                    "--index_path", idx, "--result_path", out] + extra, pin=True)
        for line in out.read_text().splitlines()[1:]:
            f = [x.strip() for x in line.split(",")]
            if len(f) < 5:
                continue
            efs, rec, ndc, ms = int(f[0]), float(f[1]), float(f[2]), float(f[3])
            b = best.get(efs)
            if b is None:
                best[efs] = {"efs": efs, "recall": rec, "ndc": ndc, "ms": ms}
            else:
                # recall and distance count are deterministic; only time moves
                assert abs(b["recall"] - rec) < 1e-6, f"{stage} efs={efs} recall moved"
                b["ms"] = min(b["ms"], ms)
    out.unlink(missing_ok=True)
    return [best[e] for e in sorted(best)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--metric", default="ip_float")
    ap.add_argument("--efC", type=int, default=EFC,
                    help="HNSW efConstruction for the bottom graph. Their "
                         "driver ships 500; the paper's other arms are built "
                         "at 200, and matching it is what makes the purple "
                         "curve a comparison of methods rather than builders.")
    ap.add_argument("--pops", default="hard,test")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--tag", default="",
                    help="suffix on the result name and the index names, so a "
                         "history-budget sweep does not overwrite itself")
    ap.add_argument("--keep-index", action="store_true")
    ap.add_argument("--out-tag", default="",
                    help="suffix on the RESULT name only. --tag also renames the "
                         "indexes, which forces a rebuild; the timed rerun wants "
                         "the same graphs searched by a different clock.")
    ap.add_argument("--timed", action="store_true",
                    help="search through search_hnsw_ngfix_timed: their search "
                         "call under cpp/hnsw_bench's clock, so the ms land on "
                         "the same axis as every other arm")
    a = ap.parse_args()

    tag = f"{a.corpus}__{a.arm}"
    itag = tag + a.tag
    for f in ("base.fbin", "hist.fbin", "hard.fbin", "hard_gt.ibin"):
        if not (DATA / f"{tag}_{f}").exists():
            raise SystemExit(f"missing {DATA/f'{tag}_{f}'}; run measure/ngfix_prep.py")
    sp = dict(__import__("numpy").load(RES / f"ngfix_split_{tag}.npz"))

    bt, nb = build_all(tag, a.metric, itag, efc=a.efC)
    out = {"corpus": a.corpus, "arm": a.arm, "library": "HNSW-NGFix* (SIGMOD'26)",
           "params": {"M": M, "MEX": MEX, "efC": a.efC,
                      "efC_AKNN": EFC_AKNN, "build_threads": 32, "K": 10},
           "efC_matched": a.efC == 200,
           "index_bytes": nb,
           "n_history": int(sp["n_history"]) if "n_history" in sp
                        else int(len(sp["hist"])),
           "history_mode": str(sp["mode"]) if "mode" in sp else "split",
           "n_test": int(len(sp["test"])),
           "n_hard": int(len(sp["hard"])), "split_seed": int(sp["split_seed"]),
           "search_threads": 1, "pinned_core": PIN_CORE if EXCLUSIVE else None,
           "timing_valid": EXCLUSIVE, "reps": 3,
           "timing_harness": (
               "search_hnsw_ngfix_timed (their searchKnn under cpp/hnsw_bench's "
               "protocol: warm-up sweep, calibration, inner repeats to 3 s, "
               "minimum of 3 regions, sweep timed whole) -- COMPARABLE with "
               "cpp/hnsw_bench"
               if a.timed else
               "search_hnsw_ngfix (gettimeofday, one pass per rep, min of three) "
               "-- NOT comparable with cpp/hnsw_bench"),
           "build_time_s": bt,
           "memory_note": (
               "index_bytes is the whole saved graph. `bottom` is an HNSW at "
               "M=16 and is the size the rest of the paper's arms are; the "
               "repaired stages add up to MEX=48 edges per node on top of it, "
               "so they are NOT at equal memory with the other arms and the "
               "ratio is recorded so the curve can be read against it."),
           "grids": {}}
    for st in a.stages.split(","):
        for pop in a.pops.split(","):
            rows = search(tag, st, pop, itag=itag, timed=a.timed)
            out["grids"].setdefault(st, {})[pop] = rows
            top = rows[-1]
            print(f"  [{tag}] {st:7s} {pop:5s} {len(rows)} points, "
                  f"deepest recall {top['recall']:.4f} at {top['ms']:.3f} ms",
                  flush=True)
    p = RES / f"ngfix_{tag}{a.tag}{a.out_tag}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p.name}", flush=True)
    if not a.keep_index:
        for st in STAGES:
            (IDX / f"{itag}_{st}.index").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
