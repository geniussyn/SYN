#!/bin/bash
# Assemble the artifact from the working trees. It copies; it does not compute.
#
# WHAT GOES IN IS WHAT THE PAPER USES. The lab tree holds 330 result files, 99
# scripts and 20 rendered figures, because measuring honestly means measuring
# things that turn out not to belong in the paper. None of that is deleted -- it
# stays in the lab -- but none of it ships either, because a reviewer who runs a
# script that has no data, or finds a collection the paper never mentions, is
# being asked to work out what is evidence and what is residue. That is the
# author's job, done here.
#
# Dropped deliberately, and where to find it if anyone asks:
#
#   DARTH / ada-ef     the early-termination baseline. Measured on all four
#                      collections, on both embeddings, and twice over after we
#                      found and the authors confirmed a feature-ordering bug in
#                      the released code. It is out of this submission, so it is
#                      out of here: the runs, the two patches against it, the
#                      four scripts that drive it and its figure panels. All of
#                      it is in the lab.
#   fig_crosslingual   the two mMARCO query sets on their own axes. Superseded:
#                      fig_strata carries them as two of its panels, so the
#                      paper shows them once. In the lab as
#                      code/fig_crosslingual.py; the measurements it reads ship,
#                      because fig_strata reads the same files.
#   anchor_locality    whether the gain is local to the 4,944 mined documents.
#                      It is not -- queries whose target set contains none of
#                      them improve 55.1% against 55.4% overall -- but the paper
#                      has no room for it. In the lab as code/anchor_locality.py
#                      and results/anchor_locality_*.json, ready if a reviewer
#                      asks how 500 queries move a million documents.
#   MS COCO            measured as a fifth collection, not in the paper.
#                      lab/results/excluded/, with the reason.
#   de ru zh ja hi id  cross-lingual screening; only ar and fr are measured
#                      end to end and only those two are claimed.
#   63 scripts         earlier campaigns, superseded figures, annotation tools.
#                      They are in the lab and most of them no longer have data.
#                      fig_rerank_pipeline is among them: the three metrics it
#                      drew on one axes are three rows of fig_rerank_strata, and
#                      the all-query stratum it showed is that figure's top line.
#                      rerank_pipeline.py, which MEASURES the pipeline, stays.
#   graph_*.npz        1.2 GB of dumped level-0 adjacency. Its only product is
#                      phi, which is already in the cell JSON and in the phi
#                      distribution cache that the figure reads.
#
# The two lists below are computed, not remembered:
#   - the modules are the transitive import closure of the entry points
#   - the result globs are what a python audit hook recorded the ten figures and
#     the three tables actually opening
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LAB="${LAB:-/root/SYN}"
TRAIN="${TRAIN:-/root/SYN-train}"

rm -rf "$HERE"/lab "$HERE"/syn "$HERE"/tests
mkdir -p "$HERE"/{lab/code,lab/cpp,lab/work,lab/results,lab/figs/final,lab/corpora,lab/tailcensor/out,lab/annot,lab/phih/scripts,syn,tests}

# ---------------------------------------------------------------- training ---
cp "$TRAIN"/syn/*.py            "$HERE"/syn/
cp "$TRAIN"/tests/*.py          "$HERE"/tests/
cp "$TRAIN"/{run.sh,requirements.txt} "$HERE"/ 2>/dev/null || true
# .gitignore, LICENSE and ANONYMITY.md are the repository's own, not copies of
# anything in the lab, so collect.sh leaves them alone; they are listed here so
# that a reader of this script knows they are expected to exist.

# ------------------------------------------------------------- measurement ---
# the entry points the paper needs, plus everything they import
MODULES="audit_timing build_mmarco encode_cache
         fig_dist fig_families fig_ivf_spread fig_mutual_rank
         fig_ngfix fig_rerank_strata fig_strata fig_sweep
         goldk_own gt hnsw_io housestyle index_hnsw ivf_spread kappa
         measured_speedup metrics ngfix_pool_table ngfix_prep oodness own_decile
         ivf_table sweep_table
         paths rerank_pipeline rerank_strata run_cell run_family
         run_ngfix screen_arm strata_cell style
         annot_agree build_stability config_table
         fig_ablation fig_encoders
         fig_sweep_ablation gold_loss"
for m in $MODULES; do
  cp "$LAB/code/$m.py" "$HERE/lab/code/" || echo "  !! code/$m.py missing"
done

cp "$LAB"/cpp/*.cpp "$LAB"/cpp/Makefile "$HERE"/lab/cpp/
cp -r "$LAB"/cpp/third_party "$HERE"/lab/cpp/       # hnswlib, header-only, MIT

# The lanes that produced the shipped results. They were written against this
# machine, so `cd /root/SYN` becomes `cd "${SYN_ROOT:-$(...)}"` on the way in and
# the two housekeeping lanes that only moved files between disks do not ship.
for w in run_all queue ngfix ngfix_pool fever_sweeps crosslingual mmarco \
         strata ablation ablation_sweep encoders tail4 tail5 tail6; do
  [ -f "$LAB/work/$w.sh" ] || continue
  sed -e 's|^cd /root/SYN$|cd "${SYN_ROOT:?set SYN_ROOT to the lab directory}"|' \
      -e 's|/root/SYN-artifact|"$ARTIFACT"|g' \
      -e 's|/workspace/SYN-train|"${SYN_TRAIN:?set SYN_TRAIN to the training tree}"|g' \
      -e 's|/root/SYN-train|"${SYN_TRAIN:?set SYN_TRAIN to the training tree}"|g' \
      -e 's|/root/SYN|"$SYN_ROOT"|g' \
      "$LAB/work/$w.sh" > "$HERE/lab/work/$w.sh"
done

# ------------------------------------------------------------------ results --
# JSON is the evidence. The npz are the two per-query arrays a figure needs:
# perquery_* for the recall distribution, and MS MARCO's ground truth and hard
# decile for the mutual-rank example.
for g in "cell_*.json" "strata_*.json" "family_*__own_M8.json" \
         "ngfix_*.json" \
         "rerank_pipeline_*.json" "rerank_pipeline_P*.json" \
         "phi1_dist_*.json" "ivf_spread_*.json" "oodness_*.json" \
         "screen_*.json" "gold_loss_*.json" "build_stability_*.json" \
         "config_table.json" "ablation_table.tex" "config_table.tex" \
         "ivf_table.tex" "sweep_table.tex" \
         "mutual_rank_panels*.json" "annot_summary.json" \
         "measured_speedup_*.json" "ngfix_pool_*.json" \
         "perquery_*.npz" "gt_msmarco-1m__*.npz" "harddecile_msmarco-1m__*.npz"; do
  for f in "$LAB"/results/$g; do
    [ -e "$f" ] || continue
    case "$(basename "$f")" in
      *mscoco*|*_hard20_*) continue ;;
      # Section 5's encoder comparison is MS MARCO only, for space. All five
      # encoders were measured on FEVER too and the ordering is the same, but
      # the figure does not read those runs, so they do not ship. The reference
      # encoder has no suffix, so its FEVER arms are not caught here -- they are
      # the paper's main FEVER measurement and belong in the repository.
      *_fever__*-mini__*|*_fever__*-e5l__*|*_fever__*-gte__*|*_fever__*-qw4__*)
        continue ;;
    esac
    cp -L "$f" "$HERE/lab/results/"
  done
done

# the withdrawn timings, with the note that condemns them
if [ -d "$LAB/results/superseded" ]; then
  cp -r "$LAB/results/superseded" "$HERE/lab/results/"
fi

# what each collection was cut from, and with which seed -- for the collections
# the paper claims, and the train-split pool NGFix and SYN both draw their
# history from. The other mMARCO languages were screened by drift only and are
# not measured end to end, so they do not ship either.
for c in msmarco-1m fever nq quora msmarco-fr msmarco-ar msmarco-1m-trainpool; do
  [ -f "$LAB/corpora/${c}_manifest.json" ] \
    && cp "$LAB/corpora/${c}_manifest.json" "$HERE/lab/corpora/"
done

# -------------------------------------------------------------------- phi_h ---
# The fragmentation index as a hardness measure, against the classical distance
# measures and against Steiner-hardness. This ran on a different machine and on
# public ANN benchmark corpora (GloVe, MSMARCO ada-002, YouTube-1M), not on the
# BEIR collections the rest of the paper uses, so its scripts take their roots
# from the environment and its corpora are not here. VALUES.txt carries every
# number in the figure and table_cost.tex the cost table.
cp "$LAB"/phih/*.md "$LAB"/phih/VALUES.txt "$LAB"/phih/*.tex \
   "$LAB"/phih/coverage_values.json "$HERE"/lab/phih/
cp "$LAB"/phih/scripts/*.py "$LAB"/phih/scripts/*.cpp "$HERE"/lab/phih/scripts/

# -------------------------------------------------------------- annotation ---
# The paper opens with "85% of hard queries arise because ...", which is a human
# judgement, not a measurement, and the only thing that substantiates it is the
# labels themselves. The guidelines the annotators worked from, the items they
# were shown, every record they returned, and the tool they used all ship, so
# the number can be recomputed and the labelling can be argued with.
cp "$LAB"/annot/*.md "$LAB"/annot/*.jsonl "$LAB"/annot/*.json "$HERE"/lab/annot/
cp "$LAB"/annot/annotate.html "$HERE"/lab/annot/ 2>/dev/null || true

# --------------------------------------------------------------- tail censor --
# A separate measurement on two much larger collections (SpaceV 100M, YouTube
# 15.7M), which motivates the paper rather than evaluating the method: it shows
# what removing the hardest queries does to a reported average. Its own scripts,
# its own curves, and the per-query arrays those curves were cut from. The base
# vectors are not here -- SpaceV's is 10 GB of int8 -- and RESULTS.md says where
# they come from and records the ground-truth audit.
cp "$LAB"/tailcensor/*.py "$LAB"/tailcensor/RESULTS.md "$HERE"/lab/tailcensor/
cp "$LAB"/tailcensor/out/*.json "$LAB"/tailcensor/out/*.npz \
   "$LAB"/tailcensor/out/summary.txt "$HERE"/lab/tailcensor/out/ 2>/dev/null || true

# ------------------------------------------------------------------ figures --
for f in fig_families fig_dist fig_strata_M16 fig_ngfix fig_rerank_strata \
         fig_tail_censor fig_hardness_measures \
         fig_sweep_ablation fig_encoders fig_mutual_rank_motivation \
         fig_mutual_rank_repair \
         figA_ngfix_all figA_rerank_all; do
  for e in pdf png; do
    [ -f "$LAB/figs/final/$f.$e" ] && cp "$LAB/figs/final/$f.$e" "$HERE/lab/figs/final/"
  done
done

# --------------------------------------------------------------------- done --
find "$HERE" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

printf '\n%s\n' "assembled $(date -u +%Y-%m-%dT%H:%MZ)"
printf '  %-14s %s\n' \
  "lab/code"    "$(ls "$HERE"/lab/code    | wc -l) modules" \
  "lab/results" "$(ls "$HERE"/lab/results | wc -l) files" \
  "lab/figs"    "$(ls "$HERE"/lab/figs/final | wc -l) files" \
  "lab/work"    "$(ls "$HERE"/lab/work    | wc -l) scripts" \
  "syn + tests" "$(ls "$HERE"/syn "$HERE"/tests | grep -c '\.py$') files"
printf '  %-14s %s\n' "total" "$(du -sh --exclude=.git "$HERE" | cut -f1)"
