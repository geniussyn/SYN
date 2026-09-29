"""Does table_coverage.tex say what coverage_values.json says?

Three of the four rows cannot be re-measured from this repository -- SIFT,
DEEP-image and OpenAI were part of an earlier nine-corpus campaign whose indexes
and ground truth are not here. What can still be checked is that the numbers
printed in the table are the numbers recorded for it, which catches the failure
mode a transcribed table actually has.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
tex = (HERE / "table_coverage.tex").read_text()
vals = json.loads((HERE / "coverage_values.json").read_text())

FIELDS = ["recall10_hard", "recall10_easy", "phi1_hard", "phi1_easy",
          "phi3_hard", "phi3_easy", "pr_phi3_eq1_hard"]
bad = []
for row in vals["rows"]:
    line = next((l for l in tex.splitlines()
                 if re.match(r"\s*" + re.escape(row["corpus"]) + r"\s*&", l)), None)
    if line is None:
        bad.append(f"{row['corpus']}: no row in the table")
        continue
    # strip LaTeX: bold wrappers, the {,} digit separator, the row terminator
    # columns, not a regex over the line: the metric cell contains "$L_2$" and
    # the digit separator is "{,}", both of which a bare number scan picks up
    cells = [c.strip() for c in line.split("&")]
    if len(cells) < 11:
        bad.append(f"{row['corpus']}: {len(cells)} columns, expected 11")
        continue

    def num(c):
        c = (c.replace("{,}", "").replace("\\textbf{", "").replace("}", "")
              .replace("\\\\", "").strip())
        return float(c)

    want = [float(row["n"]), float(row["dim"])] + [float(row[f]) for f in FIELDS]
    try:
        got = [num(cells[2]), num(cells[3])] + [num(cells[4 + i]) for i in range(7)]
    except ValueError as e:
        bad.append(f"{row['corpus']}: unparsable cell ({e})")
        continue
    if any(abs(a - b) > 1e-9 for a, b in zip(want, got)):
        bad.append(f"{row['corpus']}: table {got} vs values {want}")

if bad:
    print("  FAIL  table_coverage.tex disagrees with coverage_values.json")
    for b in bad:
        print("        " + b)
    sys.exit(1)
n = len(vals["rows"])
r = len(vals["reproducible_here"])
print(f"  ok    {n} rows match coverage_values.json "
      f"({r} of {n} re-measurable from this repository)")
