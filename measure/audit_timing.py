"""Find timed grids whose milliseconds cannot be a measurement.

A larger search budget cannot be faster. Every harness in this repository already
checks that inside one run; this checks it across everything already written,
because the check was added after some of it was measured, and because a run that
shared the machine with an unlocked build in another lane shows exactly this
signature: a step where the parameter rises and the time falls.

Small inversions at the shallow end are scheduling noise that min-of-three did
not fully remove. A drop of more than `--tol` of the previous point is not: at
nprobe 10 -> 24 the work trebles, so a run that came back three times faster was
measured against something else running.

Anything this flags is re-measured, not repaired.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT                                          # noqa: E402

RES = ROOT / "results"


def grids(d):
    """(name, [(param, ms)]) for every timed grid in one result file."""
    out = []
    for key, pk, mk in (("grid", "param", "ms_per_query"),
                        ("grid_hard", "param", "ms_per_query"),
                        ("grid", "ef", "ms_per_query_hard"),
                        ("grid_deep_hard", "ef", "ms_per_query_hard")):
        rows = d.get(key)
        if not isinstance(rows, list):
            continue
        g = [(r[pk], r[mk]) for r in rows
             if isinstance(r, dict) and pk in r and mk in r
             and "probe_for" not in r]
        if len(g) > 2:
            out.append((f"{key}.{mk}", sorted(g)))
    for name, s in (d.get("strata") or {}).items():
        g = [(r["ef"], r["ms"]) for r in s.get("grid", [])
             if "probe_for" not in r]
        if len(g) > 2:
            out.append((f"strata.{name}", sorted(g)))
    for arm, blk in (d.get("arms") or {}).items():
        rows = blk.get("rows") if isinstance(blk, dict) else None
        if isinstance(rows, list) and rows and "ms_search" in rows[0]:
            g = [(r["R"], r["ms_search"]) for r in rows]
            if len(g) > 2:
                out.append((f"rerank.{arm}.ms_search", sorted(g)))
    # `grids` is used by more than one baseline and the parameter is named
    # differently in each; take whichever is present rather than assuming one
    for st, pops in (d.get("grids") or {}).items():
        if not isinstance(pops, dict):
            continue
        for pop, rows in pops.items():
            if not isinstance(rows, list):
                continue
            pk = next((k for k in ("efs", "efSearch", "ef", "param")
                       if rows and isinstance(rows[0], dict) and k in rows[0]), None)
            mk = next((k for k in ("ms", "ms_per_query") if pk and k in rows[0]), None)
            if not (pk and mk):
                continue
            g = [(r[pk], r[mk]) for r in rows if r.get(mk) is not None]
            if len(g) > 2:
                out.append((f"{st}.{pop}", sorted(g)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.10,
                    help="a drop larger than this share of the previous point "
                         "is contention, not noise")
    ap.add_argument("--glob", default="*.json")
    a = ap.parse_args()

    flagged, clean = [], 0
    for p in sorted(RES.glob(a.glob)):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        worst = []
        for name, g in grids(d):
            for (p0, m0), (p1, m1) in zip(g, g[1:]):
                if m0 > 0 and (m0 - m1) / m0 > a.tol:
                    worst.append((name, p0, m0, p1, m1, (m0 - m1) / m0))
        if worst:
            flagged.append((p.name, sorted(worst, key=lambda x: -x[5])))
        elif grids(d):
            clean += 1

    for name, w in flagged:
        print(f"\n{name}   {len(w)} inversion(s)")
        for g, p0, m0, p1, m1, d_ in w[:4]:
            print(f"    {g:24s} {p0:6d} -> {p1:6d}   "
                  f"{m0:9.3f} -> {m1:8.3f} ms   -{d_*100:.0f}%")
    print(f"\n{len(flagged)} file(s) flagged, {clean} clean, tol={a.tol:.0%}")


if __name__ == "__main__":
    main()
