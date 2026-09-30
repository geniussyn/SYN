"""Table: the encoder ladder as five numbers, beside the figure that draws it.

Figure 9 shows the curves; this is the t@0.95 each one crosses, on the MS MARCO
hard decile at HNSW M=8. Vanilla, SYN, and the ratio.

The ratio column is what the section argues from, and it is worth reading beside
the dim column rather than down it alone. The gain is not ordered by width: the
three 1024d encoders span 12.7x, 7.3x and 6.5x, a spread as wide as the one
across the whole 384-to-2560 range, and neither the narrowest nor the widest
encoder sits at an end. What does order it is the Vanilla column -- the encoders
with the slowest untrained tails gain the most -- which is the same statement as
the paper's, that fragmentation is a property of the embedding rather than of
its dimension.

Absolute milliseconds are not comparable across rows: a 2560d distance costs two
and a half times a 1024d one, so the wide encoders are slower on both arms for a
reason that has nothing to do with the graph. The ratio divides that out, which
is why it is the column the text quotes.

FEVER was measured for all five and behaves the same way. It is not in this
table or the figure, for space.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT                                           # noqa: E402

RES = ROOT / "results"
CORPUS = "msmarco-1m"
TAG = "__own_M8"
TARGET = 0.95
# suffix, label, dimension -- ordered by width, as the paper prints them
ENCODERS = [("-mini", "MiniLM-L6", 384),
            ("", "BGE-M3", 1024),
            ("-e5l", "E5-large", 1024),
            ("-gte", "GTE-large", 1024),
            ("-qw4", "Qwen3-4B", 2560)]


def t_at(arm, target=TARGET):
    p = RES / f"family_hnswlib_{CORPUS}__{arm}{TAG}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    grid = sorted((r["ms_per_query"], r["recall"])
                  for r in d.get("grid_hard", []) if "recall" in r)
    for ms, recall in grid:
        if recall >= target:
            return ms
    return None


def main(out="encoders_table.tex"):
    rows, missing = [], []
    for suffix, label, dim in ENCODERS:
        van, syn = t_at(f"vanilla{suffix}"), t_at(f"syn{suffix}")
        if not (van and syn):
            missing.append(label)
            continue
        rows.append((label, dim, van, syn, van / syn))
    if not rows:
        print("no encoder runs measured")
        return False

    lines = [f"% MS MARCO 1M hard decile, HNSW M=8, t@{TARGET} in ms",
             "% recall is each arm against its own exact top-10"]
    for label, dim, van, syn, ratio in rows:
        lines.append(f"    {label:<10}& {dim:>4} & {van:5.2f} & {syn:4.2f} "
                     f"& ${ratio:.1f}\\times$ \\\\")
    (RES / out).write_text("\n".join(lines) + "\n")
    print(f"wrote results/{out}")
    print(f"  {'encoder':<11}{'dim':>5}{'Vanilla':>9}{'SYN':>7}{'ratio':>8}")
    for label, dim, van, syn, ratio in rows:
        print(f"  {label:<11}{dim:>5}{van:>9.2f}{syn:>7.2f}{ratio:>7.1f}x")
    if missing:
        print(f"  !! not measured: {', '.join(missing)}")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="encoders_table.tex")
    main(ap.parse_args().out)
