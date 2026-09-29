#!/bin/bash
# MS MARCO with Arabic and French queries: the two ends of the drift screen.
#
# The documents are the same 1M English passages and the same embedding files --
# msmarco-ar and msmarco-fr symlink to them -- so the graph under test is
# literally the same graph. The qrels are the same qrels, by query id. Only the
# query distribution moves, from drift 0.334 (English) to 0.280 (French) to
# 0.186 (Arabic), where 31% of queries share nothing with their own nearest
# document's neighbourhood.
#
# That makes this the sharpest form of the paper's claim available: if the
# hard-decile gain survives in Arabic, the repair is in the document geometry,
# because nothing else changed.
set -uo pipefail
cd "${SYN_ROOT:?set SYN_ROOT to the lab directory}"
export SYN_EXCLUSIVE=1
export SYN_PIN_CORE="${SYN_PIN_CORE:-127}"
AD="${SYN_TRAIN:?set SYN_TRAIN to the training tree}"/adapters/msmarco-1m/adapter_syn
LANGS="${LANGS:-ar fr}"

for L in $LANGS; do
  C="msmarco-$L"
  for A in vanilla syn; do
    if [ ! -f "emb/${C}__${A}_queries.npy" ]; then
      echo; echo "### encode $C $A (queries only; documents are shared) ###"
      if [ "$A" = vanilla ]; then
        python3 -u analysis/encode_cache.py --corpus "$C" --arm "$A" --encoder bge-m3 \
            --bs 384 --queries-only || { echo "FAILED encode $C $A"; continue; }
      else
        python3 -u analysis/encode_cache.py --corpus "$C" --arm "$A" --encoder bge-m3 \
            --bs 384 --queries-only --adapter "$AD" \
          || { echo "FAILED encode $C $A"; continue; }
      fi
    fi
    [ -f "results/gt_${C}__${A}_k10.npz" ] || \
        python3 -u analysis/gt.py --corpus "$C" --arm "$A" -k 10 || continue
    [ -f "results/harddecile_${C}__${A}.npz" ] || \
        python3 -u analysis/own_decile.py --corpus "$C" --arm "$A"
  done
  python3 -u analysis/oodness.py --corpora "$C" --arm syn || true
  for A in vanilla syn; do
    [ -f "results/strata_${C}__${A}__M16__strata_efc200.json" ] && continue
    echo; echo "### strata $C $A ###"
    python3 -u analysis/strata_cell.py --corpus "$C" --arm "$A" -M 16 --efc 200 \
      || echo "FAILED strata $C $A"
  done
done
python3 -u analysis/audit_timing.py
echo "### CROSSLINGUAL COMPLETE ###"
