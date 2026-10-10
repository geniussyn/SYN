#!/bin/bash
# Everything after the MS MARCO DARTH runs, in one lane.
set -uo pipefail
cd "${SYN_ROOT:?set SYN_ROOT to the lab directory}"
export SYN_EXCLUSIVE=1
export SYN_PIN_CORE="${SYN_PIN_CORE:-127}"
while pgrep -f "work/(queue|ngfix|darth|chain_darth)\.sh" > /dev/null; do sleep 120; done

bash work/darth_rest.sh          # the question the user asked first
bash work/ngfix_pool.sh
bash work/fever_sweeps.sh

echo; echo "### deferred: strata quora syn ###"
[ -f results/strata_quora__syn__M16__strata_efc200.json ] || \
  python3 -u measure/strata_cell.py --corpus quora --arm syn -M 16 --efc 200 \
    || echo "FAILED strata quora syn"
for S in hard20 hard P95 P99; do
  python3 -u measure/rerank_strata.py --corpus quora --arms vanilla,syn --stratum "$S" || true
done
for f in fig_strata fig_families fig_rerank_strata fig_rerank_pipeline \
         fig_ngfix fig_sweep darth_gain measured_speedup audit_timing; do
  python3 -u "code/$f.py" 2>&1 | tail -1 | sed "s/^/  [$f] /" || true
done

bash work/crosslingual.sh
echo "### TAIL COMPLETE ###"
