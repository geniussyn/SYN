#!/bin/bash
# Which part of the objective is doing the work.
#
# Six arms, each one change from the shipped configuration, all in bge-m3's
# space on MS MARCO, all measured the same way: HNSW at M=8, each arm on its own
# hard decile, against its own exact top-10.
#
#   full      the shipped adapter: qradial gradient, L = 1100   (measured)
#   nopull    --lambda-pull 0. The contrastive term alone. If the numbers
#             survive this, the boundary pull is decoration and the paper is
#             about hard-negative mining.
#   plain     --pull-grad plain. The same loss value, but the gradient may use
#             the whole document-to-document direction instead of the tangent
#             perpendicular to the query. This isolates the restriction and
#             nothing else. The adapter was trained in September; only the
#             measurement is missing.
#   uniform   --pos-select uniform. Positives drawn flat instead of by softmax
#             over the band.
#   L130, L32 the threshold halved and halved again, at the shipped gradient.
#             L is the one number the method asks a user for. The pull files
#             exist already (pull_c3 at 130, pull_h1 at 32), so these cost a
#             training run and no mining.
#
# The two L arms that exist as adapters (syn-h1, syn-c3) were trained with the
# PLAIN gradient, which confounds L with the restriction, so they are retrained
# here at qradial rather than reused.
set -uo pipefail
cd "${SYN_ROOT:?set SYN_ROOT to the lab directory}"
export SYN_EXCLUSIVE=1
export SYN_PIN_CORE="${SYN_PIN_CORE:-127}"
TRAIN="${SYN_TRAIN:?set SYN_TRAIN to the training tree}"
# syn.* lives in the training tree, not the lab; without this every `python3 -m
# syn.train` fails with ModuleNotFoundError, which is how the first attempt lost
# all four arms.
export PYTHONPATH="$TRAIN${PYTHONPATH:+:$PYTHONPATH}"
MINE=$TRAIN/work/mine/msmarco-1m
AD=$TRAIN/adapters/ablation
C=msmarco-1m

step () { echo; echo "### $* ###"; date -u "+    %H:%M UTC"; }

measure () {          # arm
  local A=$1
  [ -f "results/gt_${C}__${A}_k10.npz" ] || \
    python3 -u analysis/gt.py --corpus $C --arm "$A" -k 10 || return 1
  [ -f "results/harddecile_${C}__${A}.npz" ] || \
    python3 -u analysis/own_decile.py --corpus $C --arm "$A" || return 1
  [ -f "results/family_hnswlib_${C}__${A}__own_M8.json" ] && return 0
  step "hnsw $A"
  python3 -u analysis/run_family.py --corpus $C --arm "$A" --family hnswlib \
      --stock-arm "$A" --time-hard --tag "__own_M8" || return 1
}

# --- the arm that is already trained, and was never measured -----------------
if [ -f "emb/${C}__syn-plain_docs.npy" ]; then
  measure "syn-plain" || echo "FAILED syn-plain"
  python3 -u analysis/fig_ablation.py 2>&1 | tail -1
fi

# --- the four that need a training run --------------------------------------
# arm : pull file : extra flags.  Read line by line: `for row in $ARMS` splits
# on every space, so "--lambda-pull 0" became two loop iterations and each arm
# was trained with a truncated flag.
mkdir -p "$AD"
while IFS='|' read -r A PULL FLAGS; do
  [ -n "$A" ] || continue
  ARM="abl-$A"
  OUT="$AD/$A"

  if [ -z "$(ls -d "$OUT"/adapter_* 2>/dev/null)" ]; then
    mkdir -p "$OUT"
    PF=""; [ -n "$PULL" ] && PF="--pull-name $PULL"
    step "train $A   ${PF} ${FLAGS}"
    python3 -u -m syn.train --corpus $C --corpora-dir "$PWD/corpora" \
        --mine-dir "$MINE" --out-dir "$OUT" \
        --tag "$A" --encoder bge-m3 --steps 500 \
        --lambda-pull 1.0 --pull-grad qradial --pos-select softmax \
        $PF $FLAGS \
      || { echo "FAILED train $A"; continue; }
  fi
  ADIR=$(ls -d "$OUT"/adapter_* 2>/dev/null | head -1)
  [ -n "$ADIR" ] || { echo "FAILED: no adapter under $OUT"; continue; }

  [ -f "emb/${C}__${ARM}_docs.npy" ] || \
    python3 -u analysis/encode_cache.py --corpus $C --arm "$ARM" --encoder bge-m3 \
        --adapter "$ADIR" || { echo "FAILED encode $A"; continue; }
  measure "$ARM" || echo "FAILED hnsw $A"
  python3 -u analysis/fig_ablation.py 2>&1 | tail -1
done <<'ARMS'
nopull||--lambda-pull 0
uniform||--pos-select uniform
L130|pull_c3|
L32|pull_h1|
ARMS
echo "### ABLATION COMPLETE ###"
