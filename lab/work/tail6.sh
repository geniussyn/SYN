#!/bin/bash
# Retry: the NGFix budget sweep over the shared query pool.
#
# The first attempt (work/tail.log, "NGFix pool ...") failed in every one of its
# eight cells, and not for a reason about NGFix: /dev/shm was at 97% because the
# eight DARTH feature dumps (58 GB of train_k10_li2.csv) were still sitting
# there after their runs finished. The 500-query cell built a truncated graph
# and searched it to an empty grid; every larger cell could not even write its
# history. The dumps are deleted -- every DARTH measurement is already in
# results/darth_*.json and the tuned predictors are archived in
# results/darth_work/ -- so there are 68 GB free and the sweep can be run as it
# was meant to be.
set -uo pipefail
cd "${SYN_ROOT:?set SYN_ROOT to the lab directory}"
export SYN_EXCLUSIVE=1
export SYN_PIN_CORE="${SYN_PIN_CORE:-127}"
while pgrep -f "work/(tail4|tail5|tail7|darth_rest|fever_sweeps|crosslingual)\.sh" \
      > /dev/null; do sleep 180; done

df -h /dev/shm | tail -1
bash work/ngfix_pool.sh
for f in fig_ngfix audit_timing; do
  python3 -u "code/$f.py" 2>&1 | tail -1 | sed "s/^/  [$f] /" || true
done
echo "### POOL RETRY COMPLETE ###"
