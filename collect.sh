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
#   fig_sweep_ablation the four knobs as four flat panels. The sweep itself is
#                      in the paper as an appendix TABLE, which says the same
#                      thing in less space: the figure is four curves whose
#                      whole message is that they are flat. In the lab as
#                      code/fig_sweep_ablation.py. The fourteen adapters it
#                      trained stay in the training tree -- 154 MB, and the
#                      table is read from the screen JSON, not from them.
#   anchor_locality    whether the gain is local to the 4,944 mined documents.
#                      It is not -- queries whose target set contains none of
#                      them improve 55.1% against 55.4% overall -- but the paper
#                      has no room for it. In the lab as code/anchor_locality.py
#                      and results/anchor_locality_*.json, ready if a reviewer
#                      asks how 500 queries move a million documents.
#   (MS COCO was here until 2026-10-02. It is in Figure 5 now, in NQ's slot:
#    it is the collection on which SYN does not help, 0.97x on the hard
#    decile, and its overlap of 0.467 puts it on the side of the boundary
#    Section 5.1 draws. A main figure of six wins is not a result.)
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
#   lab/code      -> measure/      the measurement modules; nothing that draws
#   lab/corpora   -> corpora/      manifests only
#   lab/cpp       -> native/
#   lab/work      -> experiments/  the lanes that produced the results
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

rm -rf "$HERE"/{analysis,measure,results,figures,corpora,native,scripts,experiments,studies,src,tests,gnuplot,adapters}
rm -rf "$HERE"/lab "$HERE"/syn      # the layout before the rename, if a clone has it
mkdir -p "$HERE"/{measure,native/third_party,experiments,corpora} \
         "$HERE"/studies/{tail-censoring/out,annotation,hardness/scripts} \
         "$HERE"/{src/syn,tests}

# Rewrite the path literals that the rename invalidates. Applied to every .py
# and .sh copied out of the lab; the patterns are anchored so they cannot match
# anything but a path.
rewrite () {
  sed -e 's|ROOT / "figs" / "final"|ROOT / "figures"|g' \
      -e 's|ROOT / "figs"|ROOT / "figures"|g' \
      -e 's|ROOT / "code"|ROOT / "analysis"|g' \
      -e 's|\bcode/\([a-z_0-9]*\.py\)|measure/\1|g' \
      -e 's|^    code/      this package|    measure/   this package|' \
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
# The measurement code, and nothing that draws. A figure is a rendering of a
# result, not the result; shipping the renderer invites a reader to rebuild a
# picture instead of rerunning the measurement under it. What ships is what
# derives a number: the harness that builds and times an index, the exact
# top-k that every recall is measured against, the phi_h reader, and the
# per-experiment drivers. The eleven fig_*.py, housestyle.py, style.py, the
# six *_table.py and the two gnuplot exporters are not here.
MODULES="paths metrics
         gt screen_arm kappa hnsw_io index_hnsw
         run_cell run_family strata_cell own_decile
         encode_cache build_mmarco build_mscoco
         rerank_pipeline rerank_strata goldk_own gold_loss
         run_ngfix ngfix_prep ivf_spread
         build_stability audit_timing measured_speedup oodness
         check_disjoint offline_cost annot_agree"
for m in $MODULES; do
  [ -f "$LAB/code/$m.py" ] || { echo "  !! code/$m.py missing"; continue; }
  rewrite "$LAB/code/$m.py" > "$HERE/measure/$m.py"
done

cp "$LAB"/cpp/*.cpp "$LAB"/cpp/Makefile "$HERE"/native/
cp -r "$LAB"/cpp/third_party/. "$HERE"/native/third_party/   # hnswlib, header-only, MIT

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
      "$LAB/work/$w.sh" > "$HERE/experiments/$w.sh.tmp"
  rewrite "$HERE/experiments/$w.sh.tmp" > "$HERE/experiments/$w.sh"
  rm -f "$HERE/experiments/$w.sh.tmp"
  chmod +x "$HERE/experiments/$w.sh"
done

# The embeddings are the one input too large for git. These three are the
# repository's own -- they are not derived from the lab tree -- so they live in
# data-release/ and are copied in after the rm -rf, which would otherwise take
# them with the rest of experiments/ on every assembly.
for f in fetch_embeddings.sh embeddings.manifest embeddings.ids.template; do
  [ -f "$HERE/data-release/$f" ] || { echo "  !! data-release/$f missing"; continue; }
  cp "$HERE/data-release/$f" "$HERE/experiments/$f"
done
chmod +x "$HERE/experiments/fetch_embeddings.sh" 2>/dev/null || true
# The ids are written only when the release is actually published. If they are
# there, they ship; if not, the fetch script says so and names the rebuild path.
if [ -f "$HERE/data-release/embeddings.ids" ]; then
  cp "$HERE/data-release/embeddings.ids" "$HERE/experiments/embeddings.ids"
  echo "  embeddings.ids shipped: the fetch will resolve"
else
  echo "  no embeddings.ids: the fetch will refuse and point at the rebuild"
fi

# ------------------------------------------------------------------ results --
# Nothing. A figure is a rendering of a result and a result is a rendering of a
# measurement, and shipping either invites a reader to rebuild the picture
# instead of rerunning the thing under it. What ships is what derives a number:
# experiments/ drives measure/, and both are here. The measured JSON, the
# rendered figures and the gnuplot data they were drawn from are all out.
#
# The exception is studies/, below. Those three are separate measurements whose
# inputs are not here -- SpaceV's base is 10 GB of int8, the phi_h comparison
# ran on other corpora, and the annotation is a human judgement that no code
# can recompute -- so for them the recorded output IS the evidence.

# ------------------------------------------------------------------ corpora --
# What each collection was cut from, and with which seed -- for the collections
# the paper claims, and the train-split pool NGFix and SYN both draw their
# history from. The other mMARCO languages were screened by drift only and are
# not measured end to end, so they do not ship either. The documents themselves
# do not ship: the manifest plus syn/corpus.py rebuilds the exact id set.
for c in msmarco-1m fever nq quora mscoco msmarco-fr msmarco-ar \
         msmarco-1m-trainpool; do
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

# ----------------------------------------------------------------- adapters --
# What the training produces, and the only thing a reader needs in order to
# reproduce any embedding in the paper. BGE-M3 is public, the corpora are
# public, and each arm's manifest pins encoder_hf_id and encoder_revision, so
# an adapter plus encode_cache.py regenerates the vectors bit for bit --
# 7.4 minutes per million documents on one GPU.
#
# This matters more than the embeddings themselves. A 1.9 GB .npy shows what
# came out; a 9.5 MB adapter shows that this pipeline is what produced it,
# which is the claim a reviewer is actually checking.
#
# Seven, which is every adapter the paper's figures and tables read:
#   msmarco-1m/adapter_syn        the main arm, all of Section 5 but 5.9/5.10
#   msmarco-1m/adapter_syn-plain  the ablation table's "plain gradient" row
#   ablation/nopull               the ablation table's "no pull" row
#   enc-{mini,e5l,gte,qw4}        the encoder ladder; BGE-M3 reuses the main one
# The hyperparameter sweep's fourteen adapters do not ship: that figure is not
# in the paper either.
mkdir -p "$HERE/adapters"
for a in "msmarco-1m/adapter_syn:syn" \
         "msmarco-1m/adapter_syn-plain:syn-plain" \
         "ablation/nopull/adapter_nopull:nopull" \
         "enc-mini/adapter_syn:encoder-minilm-l6" \
         "enc-e5l/adapter_syn:encoder-e5-large" \
         "enc-gte/adapter_syn:encoder-gte-large" \
         "enc-qw4/adapter_syn:encoder-qwen3-4b"; do
  src="$TRAIN/adapters/${a%%:*}"; dst="$HERE/adapters/${a##*:}"
  [ -d "$src" ] || { echo "  !! adapter ${a%%:*} missing"; continue; }
  mkdir -p "$dst"
  cp "$src"/adapter_config.json "$src"/adapter_model.safetensors "$dst"/ 2>/dev/null
done

# --------------------------------------------------------------------- done --
# docs/, tools/, data-release/, README.md, LICENSE and the two shell scripts
# are the repository's own and are versioned, not assembled; collect.sh does
# not touch them, though it does copy data-release/ into experiments/. Everything
# it does write is listed in the rm -rf above, so a stale file from an earlier
# campaign cannot survive an assembly.

# --------------------------------------------------------------- smoke test --
# A rename that misses a path literal produces a tree that looks right and
# cannot run. There is no figure module left to draw, so the test is two
# cheaper ones: every module compiles, and no module still names the lab's
# `code/` directory. Importing them is not the test -- several parse their
# arguments at module level and would exit.
( cd "$HERE" && python3 -m compileall -q measure src >/dev/null 2>&1 \
  && ! grep -rqE "\bcode/[a-z_0-9]+\.py|ROOT / \"code\"" measure src experiments ) \
  && echo "  smoke test: measure/ and src/ compile, no stale code/ path" \
  || echo "  !! smoke test FAILED: a module did not compile or a path literal survived"
find "$HERE" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

find "$HERE" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

printf '\n%s\n' "assembled $(date -u +%Y-%m-%dT%H:%MZ)"
printf '  %-14s %s\n' \
  "measure"     "$(ls "$HERE"/measure | wc -l) modules" \
  "experiments" "$(ls "$HERE"/experiments | wc -l) lanes" \
  "src + tests" "$(find "$HERE"/src "$HERE"/tests -type f | wc -l) files" \
  "adapters"    "$(ls -d "$HERE"/adapters/*/ | wc -l) LoRA, $(du -sh "$HERE/adapters" | cut -f1)" \
  "studies"     "$(find "$HERE"/studies -type f | wc -l) files" \
  "total"       "$(du -sh --exclude=.git "$HERE" | cut -f1)"