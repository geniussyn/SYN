#!/bin/bash
# The last stage: the IVF counterpart of the fragmentation distribution on the
# three collections it is still missing, then redraw everything.
set -uo pipefail
cd "${SYN_ROOT:?set SYN_ROOT to the lab directory}"
export SYN_EXCLUSIVE=1
export SYN_PIN_CORE="${SYN_PIN_CORE:-127}"
while pgrep -f "work/(tail4|darth_rest|ngfix_pool|fever_sweeps|crosslingual)\.sh" \
      > /dev/null; do sleep 180; done

python3 -u code/ivf_spread.py --corpora fever,nq,quora || true
for f in fig_dist fig_phi_dist fig_ivf_spread fig_families fig_strata \
         fig_ngfix fig_sweep fig_rerank_pipeline fig_rerank_strata \
         fig_mutual_rank darth_gain measured_speedup audit_timing; do
  python3 -u "code/$f.py" 2>&1 | tail -1 | sed "s/^/  [$f] /" || true
done
echo "### ALL COMPLETE ###"
