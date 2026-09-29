"""Recompute the labelling statistic the paper's first claim rests on."""
import collections, json, sys
from pathlib import Path
d = Path(sys.argv[1] if len(sys.argv) > 1 else "lab/annot")
rec = d / "records_main_M1.jsonl"
if not rec.exists() or not (d / "GUIDELINES.md").exists():
    print("  FAIL  annot/ is missing its records or its guidelines")
    raise SystemExit(1)
rows = [json.loads(l) for l in rec.read_text().splitlines() if l.strip()]
lab = collections.Counter(r["primary_label"] for r in rows)
conf = collections.Counter(r.get("confidence") for r in rows)
top, n = lab.most_common(1)[0]
print(f"  ok    {len(rows)} records, {top} {n/len(rows):.1%}, "
      f"high confidence {conf.get('high',0)/len(rows):.0%}")
