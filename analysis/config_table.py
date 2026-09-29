"""What each build setting costs, beside what it buys.

Section 5.5 sweeps the two HNSW build parameters and reports what they do to
search time. The other half of that sweep is what they cost: the index on disk,
the time to build it, and the out-degree the builder actually achieved, which is
below maxM0 = 2M and is the number that decides how much of the graph a search
step touches. Encoding cost is per corpus, not per build, and is the same for
both arms because SYN is a LoRA merged into the encoder -- it changes the
weights, not the shape.

Everything here is read from the cell JSONs the sweep already wrote. Nothing is
rebuilt, and the build times are wall clock at SYN_BUILD_THREADS, so they are
comparable to each other and to nothing else.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT                                           # noqa: E402

RES = ROOT / "results"
CELLS = [("M", 8, 200), ("M", 16, 200), ("M", 32, 200),
         ("efC", 16, 100), ("efC", 16, 200), ("efC", 16, 300)]


def row(corpus, arm, M, efc):
    p = RES / f"cell_{corpus}__{arm}__M{M}__own_efc{efc}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {"index_bytes": d.get("index_bytes"),
            "build_time_s": d.get("build_time_s"),
            "build_threads": d.get("build_threads"),
            "out_degree": d.get("measured_out_degree"),
            "maxM0": d.get("maxM0"), "n_docs": d.get("n_docs"),
            "dim": d.get("dim")}


def encode(corpus):
    p = RES / f"encode_time_{corpus}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    k = next(iter(d))
    v = d[k]
    arms = [a for a in v if isinstance(v[a], dict) and "docs_per_s" in v[a]]
    return {"bs": v.get("bs"), "n_timed": v.get("n_timed"),
            "per_arm": {a: v[a] for a in arms}, "ratio": v.get("ratio")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="msmarco-1m,fever")
    ap.add_argument("--arms", default="vanilla,syn")
    a = ap.parse_args()

    out = {}
    for c in a.corpora.split(","):
        print(f"\n{c}")
        print(f"  {'setting':14s}{'arm':9s}{'index':>10s}{'build':>9s}"
              f"{'threads':>9s}{'out-deg':>9s}{'maxM0':>7s}")
        for knob, M, efc in CELLS:
            for arm in a.arms.split(","):
                r = row(c, arm, M, efc)
                if not r:
                    continue
                lab = f"M={M}" if knob == "M" else f"efC={efc}"
                if knob == "M":
                    lab += f" (efC={efc})"
                else:
                    lab += f" (M={M})"
                out.setdefault(c, {}).setdefault(lab, {})[arm] = r
                print(f"  {lab:14s}{arm:9s}"
                      f"{r['index_bytes']/2**30:9.2f}G"
                      f"{r['build_time_s']/60:8.1f}m"
                      f"{r['build_threads']:9d}"
                      f"{(r['out_degree'] or float('nan')):9.2f}"
                      f"{(r['maxM0'] or 0):7d}")
        e = encode(c)
        if e:
            out.setdefault(c, {})["encoding"] = e
            for arm, v in e["per_arm"].items():
                print(f"  {'encoding':14s}{arm:9s}"
                      f"{v['docs_per_s']:9.0f}/s"
                      f"{v['projected_full_corpus_s']/60:8.1f}m"
                      f"   (timed on {e['n_timed']:,} docs, bs={e['bs']})")
            if e.get("ratio"):
                print(f"  {'':14s}{'ratio':9s}{e['ratio']:9.3f}"
                      f"   SYN encodes at the same rate: it is a merged LoRA, "
                      f"not a wider model")

    # a LaTeX body for the paper's own table, so the numbers are not retyped
    tex = []
    for c in a.corpora.split(","):
        rows = [(k, v) for k, v in out.get(c, {}).items() if k != "encoding"]
        if not rows:
            continue
        tex.append(f"% {c}")
        for lab, per in rows:
            for arm in a.arms.split(","):
                r = per.get(arm)
                if not r:
                    continue
                tex.append(
                    f"    {lab.replace('=', '${=}$'):28s} & {arm:8s} & "
                    f"{r['index_bytes']/2**30:.2f} & "
                    f"{r['build_time_s']/60:.1f} & "
                    f"{(r['out_degree'] or float('nan')):.2f} & "
                    f"{r['maxM0']} \\\\")
        e = out.get(c, {}).get("encoding")
        if e:
            for arm, v in e["per_arm"].items():
                tex.append(f"% encode {arm}: {v['docs_per_s']:.0f} docs/s, "
                           f"{v['projected_full_corpus_s']/60:.1f} min per 1M "
                           f"(timed on {e['n_timed']:,}, bs={e['bs']})")
    (RES / "config_table.tex").write_text("\n".join(tex) + "\n")

    p = RES / "config_table.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p.name} and config_table.tex")
    print("\nencoding cost is per corpus, not per build setting: SYN is a LoRA\n"
          "merged into the encoder, so the forward pass has the same shape at\n"
          "every M and efC. It belongs in a footnote, not in a column.")


if __name__ == "__main__":
    main()
