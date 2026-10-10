#!/bin/bash
# One collection end to end. SYN_ROOT points at the measurement repository.
#
#   ./run.sh msmarco-1m                      train on the collection
#   ./run.sh fever --transfer <adapter-dir>  apply an adapter zero-shot
set -euo pipefail

C="${1:?usage: run.sh <collection> [--transfer <adapter>]}"; shift || true
TRANSFER=""
[ "${1:-}" = "--transfer" ] && TRANSFER="${2:?adapter directory}"

: "${SYN_ROOT:?set SYN_ROOT to the measurement repository}"
HERE="$(cd "$(dirname "$0")" && pwd)"
export SYN_TRAIN_ROOT="$HERE"
PY="${PY:-python}"
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
MINE="$HERE/work/mine/$C"
AD="$HERE/adapters/$C"
EMB="$SYN_ROOT/emb/${C}__vanilla_docs.npy"
ENC="$PY $SYN_ROOT/measure/encode_cache.py --corpus $C --encoder bge-m3"

# 1. corpus and the baseline arm
[ -f "$SYN_ROOT/corpora/${C}_docs.txt" ] || \
  $PY -m syn.corpus --corpora "$C" --n-target 1000000 --out "$SYN_ROOT/corpora"
[ -f "$EMB" ] || $ENC --arm vanilla
$PY "$SYN_ROOT/measure/gt.py" --corpus "$C" --arm vanilla -k 10

# 2. train, or take an adapter zero-shot
if [ -z "$TRANSFER" ]; then
  $PY -m syn.mine  --corpus "$C" --corpora-dir "$SYN_ROOT/corpora" \
      --train-split train --doc-emb "$EMB" --out-dir "$MINE" \
      --require-gold-in-subset --n-train 500 --k 10 --n-clusters 2048
  $PY -m syn.pull  --corpus "$C" --corpora-dir "$SYN_ROOT/corpora" \
      --mine-dir "$MINE" --doc-emb "$EMB" \
      --pull-l 1100 --pull-low 1000 --pos-tau 0.02
  $PY -m syn.train --corpus "$C" --corpora-dir "$SYN_ROOT/corpora" \
      --mine-dir "$MINE" --out-dir "$AD" --tag syn \
      --steps 500 --lambda-pull 1.0 --pos-select softmax
  TRANSFER="$AD/adapter_syn"
fi

# 3. the SYN arm, and the fp32 reference for each arm's own targets
$ENC --arm syn --adapter "$TRANSFER"
$PY "$SYN_ROOT/measure/gt.py" --corpus "$C" --arm syn -k 10

# 4. each arm on its OWN hard decile
for ARM in vanilla syn; do
  $PY "$SYN_ROOT/measure/own_decile.py" --corpus "$C" --arm "$ARM"
  $PY "$SYN_ROOT/measure/run_cell.py" --corpus "$C" --arm "$ARM" \
      -M 16 --efc 200 --stock-arm "$ARM" --tag __own_efc200 \
      --time-hard --ms-cap 64
done
