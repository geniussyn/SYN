#!/bin/bash
# NGFix given the same query pool SYN drew from, and a budget sweep over it.
#
# The split-half runs manufactured a history out of the evaluation queries
# because BEIR ships no query log. That favoured NGFix -- 3,490 per collection
# against SYN's 500 for all five -- and still left the objection that its paper
# uses millions. Both are answered here: the history comes from the collection's
# BEIR *train* split, leak-filtered by syn/mine.py, which is the pool SYN's 500
# were sampled from, and the budget is swept over nested prefixes of it. At 500
# it is literally SYN's own 500 queries.
#
# Because the history is a disjoint split, the evaluation is the FULL query set
# and the hard decile the rest of the paper uses -- these numbers sit beside
# every other table, unlike the split-half ones.
#
# Only the paper's own combined stage is swept. The three-way split (ngfix /
# rfix / both) is measured once, at the full pool.
set -uo pipefail
cd "${SYN_ROOT:?set SYN_ROOT to the lab directory}"
export SYN_EXCLUSIVE=1
export SYN_PIN_CORE="${SYN_PIN_CORE:-127}"
C=msmarco-1m
POOL="${C}-trainpool"

for A in vanilla syn; do
  [ -f "emb/${POOL}__${A}_queries.npy" ] || \
    python3 -u code/encode_cache.py --corpus "$POOL" --arm "$A" --encoder bge-m3 \
        --bs 384 --queries-only \
        $([ "$A" = syn ] && echo "--adapter "${SYN_TRAIN:?set SYN_TRAIN to the training tree}"/adapters/msmarco-1m/adapter_syn") \
      || { echo "FAILED encode pool $A"; exit 1; }
done

for A in vanilla syn; do
  FIRST=1
  for N in 500 4000 16000 59242; do
    TAG="__pool${N}"
    [ -f "results/ngfix_${C}__${A}${TAG}.json" ] && continue
    echo; echo "### NGFix pool $C $A  history=$N ###"
    # the 4 GB base vectors are written once per arm, not once per budget
    NB=$([ "$FIRST" = 1 ] && echo "" || echo "--no-base"); FIRST=0
    python3 -u code/ngfix_prep.py --corpus "$C" --arm "$A" \
        --history pool --pool-corpus "$POOL" --history-n "$N" $NB \
      || { echo "FAILED prep $A $N"; continue; }
    # the full pool also measures the stage split; the sweep points need only
    # the paper's combined stage
    ST=$([ "$N" = 59242 ] && echo "bottom,ngfix,rfix,both" || echo "bottom,both")
    python3 -u code/run_ngfix.py --corpus "$C" --arm "$A" --tag "$TAG" \
        --stages "$ST" --pops hard,test || echo "FAILED ngfix $A $N"
  done
  rm -f /dev/shm/ngfix/idx/${C}__${A}_bottom.index
  rm -f /dev/shm/ngfix/${C}__${A}_*.fbin /dev/shm/ngfix/${C}__${A}_*.ibin
done
echo "### NGFIX POOL SWEEP COMPLETE ###"
