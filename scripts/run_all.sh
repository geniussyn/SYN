#!/bin/bash
# Section 6, from the top, on SYN = the query-radial arm.
#
# One adapter, trained on 500 MS MARCO queries, applied unchanged to every
# collection. Every arm is measured on ITS OWN hard decile; recall is against
# that arm's own fp32 exact top-10, while MRR, nDCG and gold recall are against
# fixed human judgements and so compare across arms directly.
#
# Search is timed single-threaded on one reserved core under a machine-wide
# lock, through the native harness, so the milliseconds are a measurement.
# Nothing else may run while a timed stage is in flight.
#
# The previous arm -- the same objective with the pull gradient unrestricted --
# is kept as `syn-plain`; its measurements are the projection ablation.
set -uo pipefail
cd "${SYN_ROOT:?set SYN_ROOT to the lab directory}"
# Search time is the reported axis again, so every timed measurement owns the
# machine: one pinned core, single-threaded, under the machine-wide lock. Stage 0
# is GPU work and finishes before anything is timed.
export SYN_EXCLUSIVE=1
export SYN_PIN_CORE="${SYN_PIN_CORE:-127}"
COLL="${COLL:-msmarco-1m fever nq quora}"
ARMS="vanilla syn"
AD="${SYN_TRAIN:?set SYN_TRAIN to the training tree}"/adapters/msmarco-1m/adapter_syn
DEEP=3072,4096,6144,8192,12288,16384,24576,32768,49152,65536
step () { echo; echo "### $* ###"; }

# 0  the SYN arm on every collection --------------------------------------
for C in $COLL; do
  if [ ! -f "emb/${C}__syn_queries.npy" ]; then
    step "0 encode $C syn (MS MARCO adapter, zero-shot off MS MARCO)"
    python3 -u analysis/encode_cache.py --corpus "$C" --arm syn --encoder bge-m3 \
        --bs 384 --adapter "$AD" || { echo "FAILED encode $C"; continue; }
  fi
  [ -f "results/gt_${C}__syn_k10.npz" ] || \
      python3 -u analysis/gt.py --corpus "$C" --arm syn -k 10 || continue
  for A in $ARMS; do
    [ -f "results/harddecile_${C}__${A}.npz" ] || \
        python3 -u analysis/own_decile.py --corpus "$C" --arm "$A"
  done
  python3 -u analysis/screen_arm.py --corpus "$C" --arm syn -M 16 --n-queries 1000 || true
done
echo "### 0 COMPLETE ###"

# 6.1  recall, ANN MRR, nDCG, gold -- both arms on the dense budget grid ---
for C in $COLL; do
  for A in $ARMS; do
    [ -f "results/cell_${C}__${A}__M16__own_efc200.json" ] && continue
    step "6.1 cell $C $A"
    python3 -u analysis/run_cell.py --corpus "$C" --arm "$A" -M 16 --efc 200 \
        --stock-arm "$A" --tag __own_efc200 --time-hard --ef-deep "$DEEP" \
      || echo "FAILED cell $C $A"
  done
done
echo "### 6.1 COMPLETE ###"

# 6.2  index families ------------------------------------------------------
for F in hnswlib faiss-hnsw faiss-nsg vamana faiss-ivf; do
  for A in $ARMS; do
    [ -f "results/family_${F}_msmarco-1m__${A}__own.json" ] && continue
    step "6.2 family $F msmarco-1m $A"
    python3 -u analysis/run_family.py --corpus msmarco-1m --arm "$A" --family "$F" \
        --stock-arm "$A" --tag __own --time-hard || echo "FAILED family $F $A"
  done
done
for C in fever nq quora; do
  for A in $ARMS; do
    [ -f "results/family_faiss-ivf_${C}__${A}__own.json" ] && continue
    step "6.2 family faiss-ivf $C $A"
    python3 -u analysis/run_family.py --corpus "$C" --arm "$A" --family faiss-ivf \
        --stock-arm "$A" --tag __own --time-hard || echo "FAILED family ivf $C $A"
  done
done
echo "### 6.2 COMPLETE ###"

# 6.3  difficulty strata; targets measured, not interpolated ---------------
for C in $COLL; do
  for A in $ARMS; do
    [ -f "results/strata_${C}__${A}__M16__strata_efc200.json" ] && continue
    step "6.3 strata $C $A"
    python3 -u analysis/strata_cell.py --corpus "$C" --arm "$A" -M 16 --efc 200 \
      || echo "FAILED strata $C $A"
  done
done
python3 -u analysis/measured_speedup.py
echo "### 6.3 COMPLETE ###"

# 6.5  gold recall at depth, as returned and exhaustively ------------------
for C in $COLL; do
  step "6.5 gold@k at depth, $C"
  python3 -u analysis/highrecall_goldk.py --corpus "$C" --arms vanilla,syn || true
  python3 -u analysis/recall_at_k.py --corpus "$C" --arms vanilla,syn || true
done
echo "### 6.5 COMPLETE ###"

# 6.6  end to end: retrieve R, rerank, keep 10 ----------------------------
for C in $COLL; do
  [ -f "results/rerank_pipeline_all_${C}.json" ] && continue
  step "6.6 rerank $C"
  python3 -u analysis/rerank_pipeline.py --corpus "$C" --arms vanilla,syn -M 16 --efc 200 \
    || { echo "FAILED rerank $C"; continue; }
  for S in hard hard20; do
    python3 -u analysis/rerank_strata.py --corpus "$C" --arms vanilla,syn --stratum "$S" || true
  done
done
echo "### 6.6 COMPLETE ###"

# 6.4  degree and construction-budget sweep -------------------------------
for M in 8 32; do
  for A in $ARMS; do
    [ -f "results/cell_msmarco-1m__${A}__M${M}__own_efc200.json" ] && continue
    step "6.4 cell msmarco-1m $A M$M"
    python3 -u analysis/run_cell.py --corpus msmarco-1m --arm "$A" -M "$M" --efc 200 \
        --stock-arm "$A" --tag __own_efc200 --time-hard --ef-deep "$DEEP" || true
  done
done
for EFC in 100 300; do
  for A in $ARMS; do
    [ -f "results/cell_msmarco-1m__${A}__M16__own_efc${EFC}.json" ] && continue
    step "6.4 cell msmarco-1m $A efC$EFC"
    python3 -u analysis/run_cell.py --corpus msmarco-1m --arm "$A" -M 16 --efc "$EFC" \
        --stock-arm "$A" --tag "__own_efc${EFC}" --time-hard --ef-deep "$DEEP" || true
  done
done
python3 -u analysis/build_cost.py || true
echo "### 6.4 COMPLETE ###"

step "figures"
for f in figs_main fig_quality fig_strata fig_families_one fig_recall_dist \
         fig_tail fig_rerank_pipeline fig_efc fig_lowm fig_build_cost fig_sweeps; do
  python3 -u "code/$f.py" 2>&1 | tail -2 | sed "s/^/  [$f] /" || true
done
python3 -u analysis/fig_rerank_pipeline.py _hard20 2>&1 | tail -1 || true
python3 -u analysis/screen_table.py || true
echo "### SECTION 6 COMPLETE ###"
