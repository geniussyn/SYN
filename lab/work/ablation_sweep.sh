#!/bin/bash
# The four continuous knobs, swept, and read without an index.
#
# The bar figure keeps the two arms that switch a mechanism off. Everything with
# a range is a curve here, and it is read with screen_arm.py rather than a timed
# search. That is not a shortcut. recall@10 is measured against each arm's own
# exact top-10, so an arm that moves its target set to an easier one improves it
# without repairing anything -- which is what the L arms did. phi-hat_1 is
# computed from the embedding alone, under the best graph a builder with
# out-degree maxM0 could make, so there is nothing to move; screen_arm.py
# reports the exhaustive ranking cost beside it.
#
# Each knob lives in a different stage, so each needs a different prefix rerun:
#
#   lambda_pull, tau_b   train only          -- reuse the shipped mine and pull
#   tau_p                pull, then train    -- pos_prob is built in pull.py
#   L                    pull, then train    -- the band is built in pull.py
#   training queries     mine, pull, train   -- a different query sample
#
# Cost: train ~50 min, encode ~40 min, screen ~10 min, plus mining where needed.
# No index build and no timed search, so this wants the GPU, not the pinned core.
set -uo pipefail
cd "${SYN_ROOT:?set SYN_ROOT to the lab directory}"
TRAIN="${SYN_TRAIN:?set SYN_TRAIN to the training tree}"
MINE=$TRAIN/work/mine/msmarco-1m           # the shipped mine: 500 queries
AD=$TRAIN/adapters/ablation
DOC=$PWD/emb/msmarco-1m__vanilla_docs.npy
export PYTHONPATH="$TRAIN${PYTHONPATH:+:$PYTHONPATH}"
C=msmarco-1m
while pgrep -f "work/(tail12|tail13|tail14|encoders|ablation)\.sh" > /dev/null; do sleep 180; done

step () { echo; echo "### $* ###"; date -u "+    %H:%M UTC"; }

finish () {            # tag, mine dir, pull name, train flags
  local A=$1 MD=$2 PN=$3; shift 3
  local ARM="abl-$A" OUT="$AD/$A"
  if [ -z "$(ls -d "$OUT"/adapter_* 2>/dev/null)" ]; then
    mkdir -p "$OUT"
    step "train $A   pull=$PN   $*"
    python3 -u -m syn.train --corpus $C --corpora-dir "$PWD/corpora" \
        --mine-dir "$MD" --out-dir "$OUT" --tag "$A" --encoder bge-m3 \
        --steps 500 --pull-grad qradial --pos-select softmax \
        ${PN:+--pull-name $PN} "$@" \
      || { echo "FAILED train $A"; return 1; }
  fi
  local ADIR; ADIR=$(ls -d "$OUT"/adapter_* 2>/dev/null | head -1)
  [ -n "$ADIR" ] || { echo "FAILED: no adapter under $OUT"; return 1; }
  [ -f "emb/${C}__${ARM}_docs.npy" ] || \
    python3 -u code/encode_cache.py --corpus $C --arm "$ARM" --encoder bge-m3 \
        --adapter "$ADIR" || { echo "FAILED encode $A"; return 1; }
  [ -f "results/screen_${C}__${ARM}.json" ] || \
    python3 -u code/screen_arm.py --corpus $C --arm "$ARM" \
      || { echo "FAILED screen $A"; return 1; }
  python3 -u code/fig_sweep_ablation.py 2>&1 | head -1
}

# ---- lambda_pull: train-only ---------------------------------------------
for LAM in 0.3 2.0 3.0; do
  finish "lam$LAM" "$MINE" "" --lambda-pull "$LAM" || true
done

# ---- tau_p: rebuild pos_prob in pull, then train --------------------------
for TP in 0.005 0.05 0.1; do
  PN="pull_tp$TP"
  [ -f "$MINE/$PN.npz" ] || \
    python3 -u -m syn.pull --corpus $C --corpora-dir "$PWD/corpora" \
        --mine-dir "$MINE" --doc-emb "$DOC" --pull-l 1100 --pull-low 1000 \
        --pos-tau "$TP" --name "$PN" || { echo "FAILED pull tp$TP"; continue; }
  finish "tp$TP" "$MINE" "$PN" --lambda-pull 1.0 || true
done

# ---- L: the band is built in pull; the two files already exist ------------
finish "L130" "$MINE" pull_c3 --lambda-pull 1.0 || true
finish "L32"  "$MINE" pull_h1 --lambda-pull 1.0 || true

# ---- training queries: a different mine, then pull, then train ------------
for N in 50 100 250; do
  MD="$TRAIN/work/mine/${C}_n$N"
  if [ ! -f "$MD/mine.npz" ]; then
    step "mine $N queries"
    python3 -u -m syn.mine --corpus $C --corpora-dir "$PWD/corpora" \
        --train-split train --require-gold-in-subset --n-train "$N" --k 10 \
        --n-clusters 2048 --encoder bge-m3 --doc-emb "$DOC" --out-dir "$MD" \
      || { echo "FAILED mine n$N"; continue; }
  fi
  [ -f "$MD/pull.npz" ] || \
    python3 -u -m syn.pull --corpus $C --corpora-dir "$PWD/corpora" \
        --mine-dir "$MD" --doc-emb "$DOC" --pull-l 1100 --pull-low 1000 \
        --pos-tau 0.02 || { echo "FAILED pull n$N"; continue; }
  finish "n$N" "$MD" "" --lambda-pull 1.0 || true
done

# ---- the arms that already have embeddings only need screening ------------
for A in abl-nopull syn syn-plain vanilla; do
  [ -f "results/screen_${C}__${A}.json" ] && continue
  [ -f "emb/${C}__${A}_docs.npy" ] || continue
  python3 -u code/screen_arm.py --corpus $C --arm "$A" || true
done
python3 -u code/fig_sweep_ablation.py 2>&1 | tail -6
echo "### ABLATION SWEEP COMPLETE ###"
