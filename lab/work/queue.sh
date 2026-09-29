#!/bin/bash
# One queue. Nothing else runs on this machine while it does.
#
# Two lanes at once was the whole problem. The lock serialises timed regions, but
# each lane's untimed stretches -- recall arithmetic, ground truth, encoding --
# ran straight through the other lane's measurement, and until 2026-09-22 the
# lock did not cover builds at all. From here the stages are sequential, and the
# load-average guard in index_hnsw.timing_lock() reports anything that is not.
set -uo pipefail
cd "${SYN_ROOT:?set SYN_ROOT to the lab directory}"

wait_for () {
  while pgrep -f "$1" > /dev/null; do sleep 60; done
}

wait_for "work/mscoco\.sh"
wait_for "code/run_ngfix\.py"

bash work/redo_contended.sh
bash work/ngfix.sh
echo "### QUEUE COMPLETE ###"
