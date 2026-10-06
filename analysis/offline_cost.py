"""What the whole offline pipeline costs, so "lightweight" is a number.

Stages 1-3 of Section 4, end to end, for one collection. Taken from the stage
manifests and training logs rather than re-run, and reported as wall clock
beside the device each stage used -- a GPU-hour figure that silently includes
seventeen minutes of CPU k-means is not the figure anyone wants.

The base encode is listed separately because it is not SYN's cost: the
embedding has to exist to build an index at all, and SYN reads the one that is
already there. Mining, the boundary and the training are what SYN adds.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT                                           # noqa: E402

# The training tree is a sibling of this one and is not inside it, so its
# location comes from the environment. SYN_TRAIN if set, else ../SYN-train.
TRAIN = Path(os.environ.get("SYN_TRAIN", ROOT.parent / "SYN-train"))
RES = ROOT / "results"


def hhmm(s):
    return f"{s/60:.1f} min" if s < 3600 else f"{s/3600:.2f} h"


def stage_times(corpus, log):
    """(top-L mine, k-means, boundary) seconds, read off a lane log.

    The three run back to back inside one mine invocation and the log stamps
    each, so the boundaries are the stamps rather than a separate measurement.
    """
    t = {}
    txt = Path(log).read_text(errors="ignore") if Path(log).exists() else ""
    # a mine run: "### mine N queries ###" then S_q, then k-means, then pull
    for m in re.finditer(
            r"### mine (\d+) queries ###\s*\n\s*(\d\d):(\d\d) UTC(.*?)"
            r"\[(\d\d):(\d\d):(\d\d)\] S_q:(.*?)"
            r"\[(\d\d):(\d\d):(\d\d)\]\s+k-means(.*?)"
            r"\[(\d\d):(\d\d):(\d\d)\] wrote .*?pull\.npz",
            txt, re.S):
        g = m.groups()
        start = int(g[1]) * 3600 + int(g[2]) * 60
        sq = int(g[4]) * 3600 + int(g[5]) * 60 + int(g[6])
        km = int(g[8]) * 3600 + int(g[9]) * 60 + int(g[10])
        pl = int(g[12]) * 3600 + int(g[13]) * 60 + int(g[14])
        t = {"mine_topL": sq - start, "kmeans": km - sq, "boundary": pl - km}
    return t


def main(corpus="msmarco-1m", log=None):
    man = json.loads((TRAIN / "work" / "mine" / corpus / "manifest.json").read_text())
    pull = json.loads((TRAIN / "work" / "mine" / corpus / "pull_manifest.json").read_text())
    tl = json.loads((TRAIN / "adapters" / corpus / "trainlog_syn.json").read_text())
    enc = RES / f"encode_time_{corpus}.json"
    encs = None
    if enc.exists():
        d = json.loads(enc.read_text())
        v = d.get("vanilla") or next(iter(d.values()))
        if isinstance(v, dict) and v.get("docs_per_s"):
            encs = man["n_docs"] / v["docs_per_s"]

    t = stage_times(corpus, log or ROOT / "work" / "tail.log")
    rows = []
    if encs:
        rows.append(("base encode (not SYN's cost)", encs, "GPU", False))
    if t:
        rows += [("1a. top-L mining", t["mine_topL"], "GPU", True),
                 ("1b. k-means regions", t["kmeans"], "CPU", True),
                 ("2. boundary construction", t["boundary"], "CPU", True)]
    rows.append(("3. LoRA training", tl["elapsed_s"], "GPU", True))

    print(f"offline cost, {corpus}, {man['n_docs']:,} documents, "
          f"{man['n_train_queries']} training queries")
    print(f"{'stage':<32}{'wall clock':>12}{'device':>8}")
    syn_s = gpu_s = 0.0
    for name, s, dev, is_syn in rows:
        print(f"{name:<32}{hhmm(s):>12}{dev:>8}")
        if is_syn:
            syn_s += s
            if dev == "GPU":
                gpu_s += s
    print(f"{'':<32}{'':>12}{'':>8}")
    print(f"{'SYN adds, total':<32}{hhmm(syn_s):>12}")
    print(f"{'  of which on the GPU':<32}{hhmm(gpu_s):>12}   "
          f"= {gpu_s/3600:.2f} GPU-hours")
    print()
    print(f"k-means: {man['n_clusters']:,} regions, "
          f"median {man['region_size_median']:.0f} documents")
    print(f"boundary: L={pull['L']}, band {pull['L_lo']}-{pull['L']}, "
          f"{pull['n_active_pairs']:,} active pairs over "
          f"{pull['n_anchors']:,} anchors")
    print(f"training: {tl['steps']} steps, batch {tl['batch_q']} queries, "
          f"LoRA r={tl['lora_r']}, {tl['trainable_params']:,} trainable")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="msmarco-1m")
    ap.add_argument("--log", default=None)
    sys.exit(main(**vars(ap.parse_args())))
