#!/bin/bash
# Does the repair depend on the encoder, or on the geometry every encoder makes?
#
# The paper measures one encoder. This runs the whole pipeline on five more,
# spanning a factor of ten in width and three architectures, on the two
# collections that carry the sweeps. HNSW only, one figure.
#
#   all-MiniLM-L6-v2             384d   the weakest and narrowest
#   bge-m3                      1024d   the paper's encoder (already measured)
#   e5-large-v2                 1024d   prefix-conditioned, mean pooled
#   gte-large-en-v1.5           1024d   a third family at the same width
#   qwen3-embedding-4b-native   2560d   decoder, last-token pooled
#
# The Qwen arm is the point of the experiment, not padding. A wider space has
# more room to keep ten documents apart without their being neighbours, so if
# fragmentation were an artefact of a cramped 1024d space it should thin out at
# 2560. It runs last because it is also the most expensive, and because the
# cheap arms find the pipeline bugs first. An 8B / 4096d arm was planned and
# dropped: it was half the remaining wall clock and the 4B arm spans the range.
#
# ONE ADAPTER PER ENCODER, trained on MS MARCO and applied to FEVER unchanged,
# exactly as the paper's is. The adapter is a LoRA merged into the encoder, so
# it cannot be applied to cached embeddings: every arm is a full re-encode.
set -uo pipefail
cd "${SYN_ROOT:?set SYN_ROOT to the lab directory}"
export SYN_EXCLUSIVE=1
export SYN_PIN_CORE="${SYN_PIN_CORE:-127}"
TRAIN="${SYN_TRAIN:?set SYN_TRAIN to the training tree}"
AD=$TRAIN/adapters
M=16; EFC=200
# syn.* lives in the training tree, not the lab
export PYTHONPATH="$TRAIN${PYTHONPATH:+:$PYTHONPATH}"
# The network volume is quota-limited and the six encoders together would write
# ~67 GB of embeddings. They are pure intermediates -- every figure reads the
# result JSON -- so they go to shared memory and are deleted as soon as the
# arm they belong to has been measured. Only bge-m3's stay on the volume,
# because fig_mutual_rank needs them.
export SYN_EMB_DIR=/dev/shm/encemb
mkdir -p "$SYN_EMB_DIR"

# code  encoder                      cheapest first: a pipeline bug costs an
#                                    hour on mini and a day on the 4B
# gte-large-en-v1.5's own modelling code indexes out of bounds under
# transformers 5.x; thenlper/gte-large is the same family at the same width on a
# stock backbone, so the ladder keeps its GTE rung.
SET="mini:all-MiniLM-L6-v2
e5l:e5-large-v2
gte:gte-large
qw4:qwen3-embedding-4b-native"

step () { echo; echo "### $* ###"; date -u "+    %H:%M UTC"; }

for row in $SET; do
  X=${row%%:*}; E=${row#*:}
  step "encoder $X = $E"

  # 1. the untrained arm, both collections
  for C in msmarco-1m fever; do
    [ -f "$SYN_EMB_DIR/${C}__vanilla-${X}_docs.npy" ] || \
      python3 -u code/encode_cache.py --corpus "$C" --arm "vanilla-$X" \
          --encoder "$E" || { echo "FAILED encode $C vanilla-$X"; continue 2; }
    [ -f "results/gt_${C}__vanilla-${X}_k10.npz" ] || \
      python3 -u code/gt.py --corpus "$C" --arm "vanilla-$X" -k 10 || continue 2
  done

  # 2. one adapter, mined and trained on MS MARCO in THIS encoder's space
  if [ ! -d "$AD/enc-$X/adapter_syn" ]; then
    step "adapter $X"
    python3 -u -m syn.mine --corpus msmarco-1m --corpora-dir "$PWD/corpora" \
        --train-split train \
        --require-gold-in-subset --n-train 500 --k 10 --n-clusters 2048 \
        --encoder "$E" --doc-emb "$SYN_EMB_DIR/msmarco-1m__vanilla-${X}_docs.npy" \
        --mine-dir "$AD/enc-$X" \
      || { echo "FAILED mine $X"; continue; }
    python3 -u -m syn.pull --corpus msmarco-1m --corpora-dir "$PWD/corpora" \
        --mine-dir "$AD/enc-$X" \
        --doc-emb "$SYN_EMB_DIR/msmarco-1m__vanilla-${X}_docs.npy" \
        --pull-l 1100 --pull-low 1000 --pos-tau 0.02 --graph-m $M \
      || { echo "FAILED pull $X"; continue; }
    python3 -u -m syn.train --corpus msmarco-1m --corpora-dir "$PWD/corpora" \
        --mine-dir "$AD/enc-$X" \
        --out-dir "$AD/enc-$X" --tag syn --encoder "$E" \
        --steps 500 --lambda-pull 1.0 --pos-select softmax \
      || { echo "FAILED train $X"; continue; }
  fi

  # 3. the trained arm, both collections, same adapter
  for C in msmarco-1m fever; do
    [ -f "$SYN_EMB_DIR/${C}__syn-${X}_docs.npy" ] || \
      python3 -u code/encode_cache.py --corpus "$C" --arm "syn-$X" \
          --encoder "$E" --adapter "$AD/enc-$X/adapter_syn" \
        || { echo "FAILED encode $C syn-$X"; continue 2; }
    [ -f "results/gt_${C}__syn-${X}_k10.npz" ] || \
      python3 -u code/gt.py --corpus "$C" --arm "syn-$X" -k 10 || continue 2
  done

  # 4. each arm's own hard decile, then the timed HNSW curve
  for C in msmarco-1m fever; do
    for A in "vanilla-$X" "syn-$X"; do
      [ -f "results/harddecile_${C}__${A}.npz" ] || \
        python3 -u code/own_decile.py --corpus "$C" --arm "$A" || continue
      [ -f "results/family_hnswlib_${C}__${A}__own_M8.json" ] && continue
      step "hnsw $C $A"
      # run_family's -M defaults to 16 and --tag is only a label, so the
      # degree has to be passed explicitly or the file lies about its build
      python3 -u code/run_family.py --corpus "$C" --arm "$A" \
          --family hnswlib -M 8 --stock-arm "$A" --time-hard --tag "__own_M8" \
        || echo "FAILED hnsw $C $A"
    done
  done
  python3 -u code/fig_encoders.py 2>&1 | tail -1
  # the measurements are written; the 5-30 GB of vectors behind them are not
  # needed again
  rm -f "$SYN_EMB_DIR"/*__vanilla-${X}_*.npy "$SYN_EMB_DIR"/*__syn-${X}_*.npy
  df -h /dev/shm | tail -1
done
echo "### ENCODERS COMPLETE ###"
