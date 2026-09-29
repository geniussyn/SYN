"""Build one arm's OWN worst-decile stratum.

The stratum every existing cell reports is defined on vanilla and held fixed
across arms, so the baseline is measured where it is weakest and every other arm
is measured on someone else's weakness. This derives the same object per arm
instead: lowest mean ANN recall against THAT arm's own exact top-10.

Protocol is unchanged from run_cell.hard_decile -- an independently seeded index
(seed 777) that no measured cell reuses, mean recall over ef 32..512, bottom
10%. Selecting and scoring on one build would regress to the mean and flatter
every arm, which is why the selector seed and the cell seed differ.
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_cell import hard_decile                        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True)
ap.add_argument("--arm", required=True)
a = ap.parse_args()

hard, acc = hard_decile(a.corpus, ref_arm=a.arm)
print(f"[{a.corpus}/{a.arm}] own decile n={len(hard)} "
      f"selector mean recall {acc[hard].mean():.4f} (all {acc.mean():.4f})")
