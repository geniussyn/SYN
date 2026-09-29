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
# the NGFix scripts' own defaults, unchanged
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


def build_all(tag, metric, itag=None):
    """One bottom graph, then one repaired copy per stage. Returns build seconds."""
    IDX.mkdir(parents=True, exist_ok=True)
    bt = {}
    itag = itag or tag
    bottom = IDX / f"{tag}_bottom.index"      # history-independent, shared
    if not bottom.exists():
        print(f"  [{tag}] bottom graph M={M} MEX={MEX} efC={EFC}", flush=True)
        _, s = run([BIN / "build_hnsw_bottom",
                    "--base_data_path", DATA / f"{tag}_base.fbin",
                    "--metric", metric, "--result_hnsw_index_path", bottom,
                    "--M", M, "--MEX", MEX, "--efC", EFC])
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
    return bt


def search(tag, stage, pop, k=10, reps=3, itag=None):
    """The efs grid on one stratum, min of `reps` passes. Returns rows."""
    idx = IDX / f"{(itag or tag) if stage != 'bottom' else tag}_{stage}.index"
    out = DATA / f"{tag}_{stage}_{pop}.csv"
    best = {}
    for _ in range(reps):
        r, _ = run([BIN / "search_hnsw_ngfix",
                    "--test_query_path", DATA / f"{tag}_{pop}.fbin",
                    "--test_gt_path", DATA / f"{tag}_{pop}_gt.ibin",
                    "--metric", "ip_float", "--K", k,
                    "--index_path", idx, "--result_path", out], pin=True)
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
    ap.add_argument("--pops", default="hard,test")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--tag", default="",
                    help="suffix on the result name and the index names, so a "
                         "history-budget sweep does not overwrite itself")
    ap.add_argument("--keep-index", action="store_true")
    a = ap.parse_args()

    tag = f"{a.corpus}__{a.arm}"
    itag = tag + a.tag
    for f in ("base.fbin", "hist.fbin", "hard.fbin", "hard_gt.ibin"):
        if not (DATA / f"{tag}_{f}").exists():
            raise SystemExit(f"missing {DATA/f'{tag}_{f}'}; run code/ngfix_prep.py")
    sp = dict(__import__("numpy").load(RES / f"ngfix_split_{tag}.npz"))

    bt = build_all(tag, a.metric, itag)
    out = {"corpus": a.corpus, "arm": a.arm, "library": "HNSW-NGFix* (SIGMOD'26)",
           "params": {"M": M, "MEX": MEX, "efC": EFC, "efC_AKNN": EFC_AKNN,
                      "build_threads": 32, "K": 10},
           "n_history": int(sp["n_history"]) if "n_history" in sp
                        else int(len(sp["hist"])),
           "history_mode": str(sp["mode"]) if "mode" in sp else "split",
           "n_test": int(len(sp["test"])),
           "n_hard": int(len(sp["hard"])), "split_seed": int(sp["split_seed"]),
           "search_threads": 1, "pinned_core": PIN_CORE if EXCLUSIVE else None,
           "timing_valid": EXCLUSIVE, "reps": 3,
           "timing_harness": "search_hnsw_ngfix (gettimeofday, one pass per rep, "
                             "min of three) -- NOT comparable with cpp/hnsw_bench",
           "build_time_s": bt, "grids": {}}
    for st in a.stages.split(","):
        for pop in a.pops.split(","):
            rows = search(tag, st, pop, itag=itag)
            out["grids"].setdefault(st, {})[pop] = rows
            top = rows[-1]
            print(f"  [{tag}] {st:7s} {pop:5s} {len(rows)} points, "
                  f"deepest recall {top['recall']:.4f} at {top['ms']:.3f} ms",
                  flush=True)
    p = RES / f"ngfix_{tag}{a.tag}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p.name}", flush=True)
    if not a.keep_index:
        for st in STAGES:
            (IDX / f"{itag}_{st}.index").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
