"""Iso-recall speedup from measured points only.

`final_speedup.py` interpolates between the two grid points that bracket a
target. This reads `strata_cell.py`'s bisected attainment instead: for each
target each arm was driven to an ef that was actually run, and what is divided
here are two measured milliseconds. The recall each arm actually returned is
printed next to the ratio, because an arm that overshoots the target is being
charged for recall it was not asked for, and the reader has to be able to see
that rather than trust that it is small.

There is no latency budget: each arm is driven until it reaches the target, so
an arm reported as unreached could not reach it at all -- ef or nprobe ran out
of room -- and that is a property of the index, not of where the measurement was
cut. Those rows carry the deepest point measured instead of a ratio.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402

RES = ROOT / "results"
# MS COCO was measured and is not in the paper; see results/excluded/
COLLS = ["msmarco-1m", "fever", "nq", "quora"]
STRATA = [("all", "all queries"), ("P80", "hard 20%"), ("P90", "hard decile")]
OVER = 64.0   # annotation only: a point this slow is outside the serving regime


def load(corpus, arm, M=16, efc=200):
    p = RES / f"strata_{corpus}__{arm}__M{M}__strata_efc{efc}.json"
    return json.loads(p.read_text()) if p.exists() else None


def cell(d, stratum, target):
    s = (d.get("strata") or {}).get(stratum)
    if not s:
        return None
    for t in s.get("targets") or []:
        if abs(t["target"] - target) < 1e-9:
            return t
    return None


def main(M=16, efc=200):
    rows = []
    for c in COLLS:
        dv, ds = load(c, "vanilla", M, efc), load(c, "syn", M, efc)
        if not (dv and ds):
            print(f"[{c}] no strata file yet")
            continue
        for key, label in STRATA:
            for R in (0.90, 0.95, 0.99):
                tv, ts = cell(dv, key, R), cell(ds, key, R)
                if not (tv and ts):
                    continue
                r = {"collection": c, "stratum": key, "stratum_label": label,
                     "target": R}
                if tv["reached"] and ts["reached"]:
                    r.update(vanilla_ms=tv["ms"], syn_ms=ts["ms"],
                             vanilla_ef=tv["ef"], syn_ef=ts["ef"],
                             vanilla_recall=tv["recall"], syn_recall=ts["recall"],
                             budget_ratio=tv["ef"] / ts["ef"],
                             speedup=tv["ms"] / ts["ms"], kind="measured")
                elif ts["reached"] and not tv["reached"]:
                    r.update(syn_ms=ts["ms"], syn_ef=ts["ef"],
                             syn_recall=ts["recall"],
                             vanilla_deepest_ms=tv["deepest_ms"],
                             vanilla_deepest_recall=tv["deepest_recall"],
                             vanilla_stopped_by=tv["stopped_by"],
                             kind="vanilla_unreached")
                elif tv["reached"] and not ts["reached"]:
                    r.update(vanilla_ms=tv["ms"], vanilla_ef=tv["ef"],
                             vanilla_recall=tv["recall"],
                             syn_deepest_ms=ts["deepest_ms"],
                             syn_deepest_recall=ts["deepest_recall"],
                             syn_stopped_by=ts["stopped_by"],
                             kind="syn_unreached")
                else:
                    r.update(kind="neither_reached")
                rows.append(r)

    out = {"M": M, "efConstruction": efc,
           "primary_axis": "search budget (ef); the ef ratio is the headline. "
                           "ms is recorded but the run was not exclusive, so it "
                           "is indicative, not a measurement",
           "method": "measured attainment: ef grown until the target is reached, "
                     "then bisected; no interpolation, no latency budget",
           "rows": rows}
    (RES / f"measured_speedup_M{M}_efc{efc}.json").write_text(json.dumps(out, indent=2))

    for key, label in STRATA:
        sel = [r for r in rows if r["stratum"] == key]
        if not sel:
            continue
        print(f"\n### {label}")
        print(f"{'collection':<14}{'R':>6}{'V ef':>8}{'SYN ef':>8}{'ef ratio':>10}"
              f"{'(ms ratio)':>12}   recall (V / S)")
        for r in sel:
            if r["kind"] == "measured":
                print(f"{r['collection']:<14}{r['target']:>6.2f}"
                      f"{r['vanilla_ef']:>8d}{r['syn_ef']:>8d}"
                      f"{r['budget_ratio']:>9.1f}x"
                      f"{r['speedup']:>11.1f}x   "
                      f"{r['vanilla_recall']:.4f} / {r['syn_recall']:.4f}")
            elif r["kind"] == "vanilla_unreached":
                print(f"{r['collection']:<14}{r['target']:>6.2f}"
                      f"{'never':>12}{r['syn_ms']:>10.3f}{'--':>10}   "
                      f"(V deepest {r['vanilla_deepest_recall']:.4f} at "
                      f"{r['vanilla_deepest_ms']:.1f} ms, {r['vanilla_stopped_by']})")
            elif r["kind"] == "syn_unreached":
                print(f"{r['collection']:<14}{r['target']:>6.2f}"
                      f"{r['vanilla_ms']:>12.3f}{'never':>10}{'--':>10}   "
                      f"(S deepest {r['syn_deepest_recall']:.4f})")
            else:
                print(f"{r['collection']:<14}{r['target']:>6.2f}"
                      f"{'--':>12}{'--':>10}{'--':>10}   neither arm reached")
    print(f"\nwrote measured_speedup_M{M}_efc{efc}.json")


if __name__ == "__main__":
    main()
