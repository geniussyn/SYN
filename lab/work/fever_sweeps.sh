#!/bin/bash
# The degree and construction-budget sweeps on FEVER, to match MS MARCO.
#
# MS MARCO has M in {8,16,32} at efC=200 and efC in {100,200,300} at M=16.
# FEVER has only the centre point, so the sweep figure can draw one collection.
# These eight cells give it the second.
set -uo pipefail
cd "${SYN_ROOT:?set SYN_ROOT to the lab directory}"
export SYN_EXCLUSIVE=1
export SYN_PIN_CORE="${SYN_PIN_CORE:-127}"
DEEP=3072,4096,6144,8192,12288,16384,24576,32768,49152,65536

cell () {  # arm, M, efC
  local A=$1 M=$2 E=$3
  [ -f "results/cell_fever__${A}__M${M}__own_efc${E}.json" ] && return
  echo; echo "### cell fever $A M$M efC$E ###"
  python3 -u code/run_cell.py --corpus fever --arm "$A" -M "$M" --efc "$E" \
      --stock-arm "$A" --tag "__own_efc${E}" --time-hard --ef-deep "$DEEP" \
    || echo "FAILED cell fever $A M$M efC$E"
}

for A in vanilla syn; do cell "$A" 8 200; done      # degree sweep
for A in vanilla syn; do cell "$A" 32 200; done
for A in vanilla syn; do cell "$A" 16 100; done     # construction budget
for A in vanilla syn; do cell "$A" 16 300; done
python3 -u code/fig_sweep.py 2>&1 | tail -1
echo "### FEVER SWEEPS COMPLETE ###"
