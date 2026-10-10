"""Are the training queries disjoint from the evaluation queries?

The question a reviewer asks first, answered by intersecting the two id sets
rather than by argument. It is not only a split separation: the training
queries come from MS MARCO's `train` split and the evaluation queries from
`dev`, but a shared split is not the only way a query can leak, so mine.py
applies three further filters and this reports what each one removed.

  L1  the training query shares a judged document with an evaluation query
  L2  the training query's id is itself an evaluation gold document
  L3  the training query's text equals an evaluation gold document

L3 is the one that matters on Quora, where queries and documents are both
questions and dev and test can be near-duplicates with disjoint ids.

Exit status is 0 when the intersection is empty, 1 when it is not, so this can
be a one-line check in CI or in a reviewer's terminal.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT                                           # noqa: E402

# The training tree is a sibling of this one and is not inside it, so its
# location comes from the environment. SYN_TRAIN if set, else ../SYN-train.
TRAIN = Path(os.environ.get("SYN_TRAIN", ROOT.parent / "SYN-train"))


def main(corpus="msmarco-1m", mine_dir=None):
    mine = Path(mine_dir) if mine_dir else TRAIN / "work" / "mine" / corpus
    npz, man = mine / "mine.npz", mine / "manifest.json"
    qfile = ROOT / "corpora" / f"{corpus}_queries.jsonl"
    for p in (npz, man, qfile):
        if not p.exists():
            print(f"missing: {p}")
            return 2

    train = {str(x) for x in np.load(npz, allow_pickle=True)["train_qids"]}
    ev = {str(json.loads(l)["qid"])
          for l in qfile.read_text().splitlines() if l.strip()}
    m = json.loads(man.read_text())
    inter = train & ev

    print(f"corpus                {corpus}")
    print(f"training split        {m.get('train_split')}   "
          f"evaluation split {m.get('eval_split')}")
    print(f"training queries      {len(train):,}")
    print(f"evaluation queries    {len(ev):,}")
    print(f"INTERSECTION          {len(inter):,}"
          + ("" if not inter else f"   {sorted(inter)[:8]}"))
    print()
    print("the pool the 500 were drawn from, and what was excluded first:")
    print(f"  train-split queries           {m.get('n_train_before_filter'):>9,}")
    print(f"  with a gold doc in the subset {m.get('n_eligible'):>9,}")
    print(f"  L1 shares a judged document   {m.get('leak_L1_shared_judgement'):>9,}")
    print(f"  L2 qid is an eval gold doc    {m.get('leak_L2_qid_is_gold_doc'):>9,}")
    print(f"  L3 qtext is an eval gold doc  {m.get('leak_L3_qtext_is_gold_doc'):>9,}")
    print(f"  excluded by the union         {m.get('n_leaked_excluded'):>9,}")
    print(f"  usable pool                   {m.get('n_pool_after_all_filters'):>9,}")
    print(f"  sampled uniformly, seed {m.get('seed')}  -> {m.get('n_train_queries')}")
    print()
    if inter:
        print("FAIL: a training query is also an evaluation query")
        return 1
    print("ok: no evaluation query was trained on")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="msmarco-1m")
    ap.add_argument("--mine-dir", default=None)
    sys.exit(main(**vars(ap.parse_args())))
