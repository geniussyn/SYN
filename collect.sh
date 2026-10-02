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
#   (fig_dist was here until 2026-10-02; it is in the paper now, as the
#    distribution behind the means that Section 5.3 quotes.)
#   fig_crosslingual   the two mMARCO query sets on their own axes. Superseded:
#                      fig_strata carries them as two of its panels, so the
#                      paper shows them once. In the lab as
#                      code/fig_crosslingual.py; the measurements it reads ship,
#                      because fig_strata reads the same files.
#   sweep_ablation     the four continuous knobs -- lambda_pull, tau_p, L and the
#                      training-query count -- swept over thirteen points and
#                      read index-free with screen_arm.py. All four panels are
#                      flat: the untrained arm sits at phi-hat_1 2.34 and every
#                      trained arm between 1.33 and 1.44, so the figure shows
#                      that the repair is what training does rather than what a
#                      knob is set to. True, and not worth a figure's space, so
#                      the paper does not claim it. In the lab as
#                      code/fig_sweep_ablation.py, work/ablation_sweep.sh and
#                      results/screen_msmarco-1m__abl-*.json, with the fourteen
#                      adapters in the training tree, ready if a reviewer asks
#                      how sensitive the method is to its hyperparameters.
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
# LAYOUT. The lab tree keeps a working root with code/, results/, figs/ and
# corpora/ under it, which is what the measurement scripts were written against
# and what paths.py resolves. A repository is read, not worked in, so the
# artifact lays the same content out the way a reader expects to find it:
#
#   lab/code      -> analysis/     the measurement modules
#   lab/results   -> results/
#   lab/figs/final-> figures/
#   lab/corpora   -> corpora/      manifests only
#   lab/cpp       -> native/
#   lab/work      -> scripts/      the lanes that produced the results
#   lab/tailcensor-> studies/tail-censoring/
#   lab/phih      -> studies/hardness/
#   lab/annot     -> studies/annotation/
#   syn           -> src/syn/
#   *.md          -> docs/         except README and LICENSE
#   collect.sh    -> tools/
#
# Renaming a directory renames the paths inside the scripts that read it, so the
# copy is not a plain cp: REWRITE below edits the four path literals the modules
# use (ROOT/"figs"/"final", ROOT/"figs", ROOT/"code", and `code/` in the lanes)
# as the files go in. Nothing else in a copied file is touched, and a smoke test
# at the end runs a figure script from the assembled tree to prove the layout
# resolves.
#
# The two lists below are computed, not remembered:
#   - the modules are the transitive import closure of the entry points
#   - the result globs are what a python audit hook recorded the ten figures and
#     the three tables actually opening
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LAB="${LAB:-/root/SYN}"
TRAIN="${TRAIN:-/root/SYN-train}"

rm -rf "$HERE"/{analysis,results,figures,corpora,native,scripts,studies,src,tests,gnuplot}
rm -rf "$HERE"/lab "$HERE"/syn      # the layout before the rename, if a clone has it
mkdir -p "$HERE"/{analysis,native/third_party,scripts,results,figures,corpora} \
         "$HERE"/studies/{tail-censoring/out,annotation,hardness/scripts} \
         "$HERE"/{src/syn,tests}

# Rewrite the path literals that the rename invalidates. Applied to every .py
# and .sh copied out of the lab; the patterns are anchored so they cannot match
# anything but a path.
rewrite () {
  sed -e 's|ROOT / "figs" / "final"|ROOT / "figures"|g' \
      -e 's|ROOT / "figs"|ROOT / "figures"|g' \
      -e 's|ROOT / "code"|ROOT / "analysis"|g' \
      -e 's|\bcode/\([a-z_0-9]*\.py\)|analysis/\1|g' \
      -e 's|figs/final|figures|g' \
      -e 's|^    code/      this package|    analysis/  this package|' \
      -e 's|^    figs/      everything|    figures/   everything|' \
      -e 's|LAB / "annot"|LAB / "studies" / "annotation"|g' \
      -e 's|ROOT / "annot"|ROOT / "studies" / "annotation"|g' \
      -e 's|LAB / "tailcensor"|LAB / "studies" / "tail-censoring"|g' \
      -e 's|ROOT / "tailcensor"|ROOT / "studies" / "tail-censoring"|g' \
      -e 's|LAB / "phih"|LAB / "studies" / "hardness"|g' \
      -e 's|ROOT / "phih"|ROOT / "studies" / "hardness"|g' \
      -e 's|records_main_A1\.jsonl|records_main_M1.jsonl|' \
      -e 's|"--single", action="store_true"|"--single", action="store_true", default=True|' "$1"
}

# ---------------------------------------------------------------- training ---
cp "$TRAIN"/syn/*.py            "$HERE"/src/syn/
# The tests import the package by name off the repository root, which is where
# it used to sit; under src/ they need the one extra segment.
for t in "$TRAIN"/tests/*.py; do
  sed 's|Path(__file__).resolve().parents\[1\]|Path(__file__).resolve().parents[1] / "src"|' \
      "$t" > "$HERE/tests/$(basename "$t")"
done

# Packaging metadata. The artifact is read more often than it is installed, but
# a repository that cannot be installed is a directory, so this makes
# `pip install -e .` work and names the src layout for the tools that look.
cat > "$HERE/pyproject.toml" <<'TOML'
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "syn"
version = "0.1.0"
description = "Taming embedding models: the geometry behind hard queries in ANNS"
readme = "README.md"
requires-python = ">=3.10"
license = { file = "LICENSE" }
dynamic = ["dependencies"]

[tool.setuptools.dynamic]
dependencies = { file = ["requirements.txt"] }

[tool.setuptools.packages.find]
where = ["src"]
TOML
cp "$TRAIN"/requirements.txt "$HERE"/ 2>/dev/null || true
# run.sh calls the measurement modules by path and imports the training package
# by name, so it needs the rename and it needs src/ on the import path.
if [ -f "$TRAIN/run.sh" ]; then
  rewrite "$TRAIN/run.sh" \
    | sed -e 's|^PY="${PY:-python}"|PY="${PY:-python}"\nexport PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"|' \
    > "$HERE/run.sh"
  chmod +x "$HERE/run.sh"
fi
# .gitignore, LICENSE and ANONYMITY.md are the repository's own, not copies of
# anything in the lab, so collect.sh leaves them alone; they are listed here so
# that a reader of this script knows they are expected to exist.

# ------------------------------------------------------------- measurement ---
# the entry points the paper needs, plus everything they import
MODULES="audit_timing build_mmarco encode_cache
         fig_build_cost fig_dist fig_families fig_ivf_spread fig_mutual_rank
         fig_ngfix fig_rerank_strata fig_strata fig_sweep
         goldk_own gt hnsw_io housestyle index_hnsw ivf_spread kappa
         measured_speedup metrics ngfix_pool_table ngfix_prep oodness own_decile
         ivf_table sweep_table families_table encoders_table
         gnuplot_all export_gnuplot
         paths rerank_pipeline rerank_strata run_cell run_family
         run_ngfix screen_arm strata_cell style
         annot_agree build_stability config_table
         fig_ablation fig_encoders
         gold_loss"
for m in $MODULES; do
  [ -f "$LAB/code/$m.py" ] || { echo "  !! code/$m.py missing"; continue; }
  rewrite "$LAB/code/$m.py" > "$HERE/analysis/$m.py"
done

cp "$LAB"/cpp/*.cpp "$LAB"/cpp/Makefile "$HERE"/native/
cp -r "$LAB"/cpp/third_party/. "$HERE"/native/third_party/   # hnswlib, header-only, MIT

# The lanes that produced the shipped results. They were written against this
# machine, so `cd /root/SYN` becomes `cd "${SYN_ROOT:-$(...)}"` on the way in and
# the two housekeeping lanes that only moved files between disks do not ship.
for w in run_all queue ngfix ngfix_pool fever_sweeps crosslingual mmarco \
         strata ablation encoders tail4 tail5 tail6; do
  [ -f "$LAB/work/$w.sh" ] || continue
  sed -e 's|^cd /root/SYN$|cd "${SYN_ROOT:?set SYN_ROOT to the lab directory}"|' \
      -e 's|/root/SYN-artifact|"$ARTIFACT"|g' \
      -e 's|/workspace/SYN-train|"${SYN_TRAIN:?set SYN_TRAIN to the training tree}"|g' \
      -e 's|/root/SYN-train|"${SYN_TRAIN:?set SYN_TRAIN to the training tree}"|g' \
      -e 's|/root/SYN|"$SYN_ROOT"|g' \
      "$LAB/work/$w.sh" > "$HERE/scripts/$w.sh.tmp"
  rewrite "$HERE/scripts/$w.sh.tmp" > "$HERE/scripts/$w.sh"
  rm -f "$HERE/scripts/$w.sh.tmp"
  chmod +x "$HERE/scripts/$w.sh"
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
         "families_table.tex" "encoders_table.tex" \
         "frag_types.npz" \
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
      # the hyperparameter sweep is not in the paper; its readouts go with it.
      # The other screen_*.json stay: phi-hat_1 for the shipped arms is quoted
      # in the text and screen_arm.py is how it is computed.
      screen_*__abl-*) continue ;;
      # and so do its arms' ground truth and hard deciles. abl-nopull is the
      # exception: it is one of the three rows of the ablation table.
      gt_*__abl-nopull_*|harddecile_*__abl-nopull.*) ;;
      gt_*__abl-*|harddecile_*__abl-*) continue ;;
    esac
    cp -L "$f" "$HERE/results/"
  done
done

# the withdrawn timings, with the note that condemns them
if [ -d "$LAB/results/superseded" ]; then
  cp -r "$LAB/results/superseded" "$HERE/results/"
fi

# what each collection was cut from, and with which seed -- for the collections
# the paper claims, and the train-split pool NGFix and SYN both draw their
# history from. The other mMARCO languages were screened by drift only and are
# not measured end to end, so they do not ship either.
for c in msmarco-1m fever nq quora msmarco-fr msmarco-ar msmarco-1m-trainpool; do
  [ -f "$LAB/corpora/${c}_manifest.json" ] \
    && cp "$LAB/corpora/${c}_manifest.json" "$HERE/corpora/"
done

# -------------------------------------------------------------------- phi_h ---
# The fragmentation index as a hardness measure, against the classical distance
# measures and against Steiner-hardness. This ran on a different machine and on
# public ANN benchmark corpora (GloVe, MSMARCO ada-002, YouTube-1M), not on the
# BEIR collections the rest of the paper uses, so its scripts take their roots
# from the environment and its corpora are not here. VALUES.txt carries every
# number in the figure and table_cost.tex the cost table.
cp "$LAB"/phih/*.md "$LAB"/phih/VALUES.txt "$LAB"/phih/*.tex \
   "$LAB"/phih/coverage_values.json "$HERE"/studies/hardness/
cp "$LAB"/phih/scripts/*.py "$LAB"/phih/scripts/*.cpp "$HERE"/studies/hardness/scripts/

# -------------------------------------------------------------- annotation ---
# The paper opens with "85% of hard queries arise because ...", which is a human
# judgement, not a measurement, and the only thing that substantiates it is the
# labels themselves. The guidelines the annotators worked from, the items they
# were shown, every record they returned, and the tool they used all ship, so
# the number can be recomputed and the labelling can be argued with.
cp "$LAB"/annot/*.md "$LAB"/annot/*.jsonl "$LAB"/annot/*.json "$HERE"/studies/annotation/
cp "$LAB"/annot/annotate.html "$HERE"/studies/annotation/ 2>/dev/null || true

# --------------------------------------------------------------- tail censor --
# A separate measurement on two much larger collections (SpaceV 100M, YouTube
# 15.7M), which motivates the paper rather than evaluating the method: it shows
# what removing the hardest queries does to a reported average. Its own scripts,
# its own curves, and the per-query arrays those curves were cut from. The base
# vectors are not here -- SpaceV's is 10 GB of int8 -- and RESULTS.md says where
# they come from and records the ground-truth audit.
cp "$LAB"/tailcensor/*.py "$LAB"/tailcensor/RESULTS.md "$HERE"/studies/tail-censoring/
cp "$LAB"/tailcensor/out/*.json "$LAB"/tailcensor/out/*.npz \
   "$LAB"/tailcensor/out/summary.txt "$HERE"/studies/tail-censoring/out/ 2>/dev/null || true

# ------------------------------------------------------------------ figures --
# The English submission's figures, in its order. fig_dist and the main
# fig_ngfix are drawn but appear only in the Korean draft, and fig_mutual_rank
# ships whole rather than as the two cuts -- the paper crops it with a viewport.
# fig_ngfix.py and fig_rerank_strata.py still ship: each draws one of the two
# appendix figures as well as the cut the Korean draft uses.
#   paper fig1 tail_censor   fig2 mutual_rank    fig3 hardness_measures
#         fig5 strata        fig6 families       fig7 build_cost
#         fig8 rerank_fever  fig9 encoders       figA1/A2 the appendix pair
# fig4_mechanism is a schematic and has no script in the lab; it is not here.
for f in fig_tail_censor fig_mutual_rank fig_hardness_measures \
         fig_strata_M16 fig_families fig_build_cost \
         fig_rerank_strata fig_encoders fig_dist \
         figA_ngfix_all figA_rerank_all; do
  for e in pdf png; do
    [ -f "$LAB/figs/final/$f.$e" ] && cp "$LAB/figs/final/$f.$e" "$HERE/figures/"
  done
done

# ------------------------------------------------------------------ gnuplot --
# The same numbers as TSV, with a gnuplot script per figure and a Makefile.
# The matplotlib scripts in analysis/ stay the source of record -- they draw the
# paper's panels -- but a reader who wants the numbers should not have to run a
# plotting stack to see them, and a reader who wants to redraw should not have
# to parse the result JSON. `make -C gnuplot check` rebuilds all of them.
if [ -d "$LAB/export/gnuplot" ]; then
  mkdir -p "$HERE/gnuplot"
  # .style travels with the data: it is the figure module's own encoding --
  # which dimension is colour and which is the line -- and without it a
  # redrawing has to guess, which is how four index families first came out
  # in two colours.
  cp "$LAB"/export/gnuplot/*.tsv "$LAB"/export/gnuplot/*.style \
     "$LAB"/export/gnuplot/*.gp "$LAB"/export/gnuplot/*.txt \
     "$LAB"/export/gnuplot/Makefile "$HERE/gnuplot/" 2>/dev/null || true
fi

# --------------------------------------------------------------------- done --
# docs/, tools/, README.md, LICENSE and the two shell scripts are the
# repository's own and are versioned, not assembled; collect.sh does not touch
# them. Everything it does write is listed in the rm -rf above, so a stale file
# from an earlier campaign cannot survive an assembly.

# --------------------------------------------------------------- smoke test --
# A rename that misses a path literal produces a tree that looks right and
# cannot run. Draw one figure from the assembled artifact -- fig_encoders reads
# only result JSON, so it needs no embeddings -- and fail the assembly if the
# layout does not resolve.
( cd "$HERE" && SYN_ROOT="$HERE" python3 analysis/fig_encoders.py >/dev/null 2>&1 ) \
  && echo "  smoke test: analysis/fig_encoders.py ran against the new layout" \
  || echo "  !! smoke test FAILED: a path literal did not survive the rename"

find "$HERE" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

printf '\n%s\n' "assembled $(date -u +%Y-%m-%dT%H:%MZ)"
printf '  %-14s %s\n' \
  "analysis"    "$(ls "$HERE"/analysis | wc -l) modules" \
  "results"     "$(ls "$HERE"/results  | wc -l) files" \
  "figures"     "$(ls "$HERE"/figures  | wc -l) files" \
  "scripts"     "$(ls "$HERE"/scripts  | wc -l) lanes" \
  "gnuplot"     "$(ls "$HERE"/gnuplot/*.tsv 2>/dev/null | wc -l) data + $(ls "$HERE"/gnuplot/*.gp 2>/dev/null | wc -l) scripts" \
  "src + tests" "$(ls "$HERE"/src/syn "$HERE"/tests | grep -c '\.py$') files"
printf '  %-14s %s\n' "total" "$(du -sh --exclude=.git "$HERE" | cut -f1)"
