#!/bin/bash
# HNSW-NGFix* (SIGMOD'26) alone and composed with SYN.
#
# Their code, their defaults (M=16, MEX=48, efC=500, efC_AKNN=1500, K=10), their
# search binary. Ours from the outside only: the pinned core, the machine-wide
# lock, min of three passes, and the split that keeps the historical queries away
# from the ones every arm is judged on.
#
# One arm at a time: the base vectors alone are 4 GB and the four indexes it
# builds are larger, so nothing is kept once its numbers are written.
set -uo pipefail
cd "${SYN_ROOT:?set SYN_ROOT to the lab directory}"
export SYN_EXCLUSIVE=1
export SYN_PIN_CORE="${SYN_PIN_CORE:-127}"
COLL="${COLL:-msmarco-1m fever nq quora}"

for C in $COLL; do
  for A in vanilla syn; do
    [ -f "results/ngfix_${C}__${A}.json" ] && continue
    echo; echo "### NGFix $C $A ###"
    python3 -u analysis/ngfix_prep.py --corpus "$C" --arm "$A" \
      || { echo "FAILED prep $C $A"; continue; }
    python3 -u analysis/run_ngfix.py --corpus "$C" --arm "$A" \
      || echo "FAILED ngfix $C $A"
    rm -f /dev/shm/ngfix/${C}__${A}_*.fbin /dev/shm/ngfix/${C}__${A}_*.ibin
    rm -f /dev/shm/ngfix/idx/${C}__${A}_*.index
  done
done
echo "### NGFIX COMPLETE ###"
