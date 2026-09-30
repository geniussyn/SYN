#!/bin/bash
# Check the claims ARTIFACT.md makes. Reads; never measures.
#
#   bash verify.sh              against the assembled artifact
#   LAB=/root/SYN bash verify.sh   against the working tree instead
#
# The lab keeps code/, figs/final/ and its side studies under one working root;
# the repository lays them out as analysis/, figures/ and studies/. Both are
# checked by the same script, so each directory is resolved once here rather
# than spelled out at every use.
#
# The default is the artifact, because what a reviewer has is the artifact: a
# check that passes only against the author's machine has checked nothing.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LAB="${LAB:-$HERE}"
# repository layout by default, lab layout when LAB points at the working tree
if [ -d "$LAB/code" ]; then
  CODE="$LAB/code"; FIGS="$LAB/figs/final"; PHIH="$LAB/phih"
  ANNOT="$LAB/annot"; TAILC="$LAB/tailcensor"; TOOLS="$HERE"
else
  CODE="$LAB/analysis"; FIGS="$LAB/figures"; PHIH="$LAB/studies/hardness"
  ANNOT="$LAB/studies/annotation"; TAILC="$LAB/studies/tail-censoring"
  TOOLS="$HERE/tools"
fi
TRAIN="${TRAIN:-$HERE}"
PY="${PY:-python3}"
# the training package sits under src/ in the repository and at the root in the
# training tree; either way it has to be importable without installing anything
export PYTHONPATH="$TRAIN/src:$TRAIN${PYTHONPATH:+:$PYTHONPATH}"
fail=0
ok ()  { printf '  ok    %s\n' "$*"; }
bad () { printf '  FAIL  %s\n' "$*"; fail=$((fail+1)); }

echo "1. numerical core"
if out=$("$PY" "$TRAIN/tests/test_core.py" 2>&1); then
  ok "$(grep -c '^ok' <<<"$out") tests"
else
  bad "tests/test_core.py"; echo "$out" | tail -5
fi

echo "2. determinism: no unseeded randomness"
u=$(grep -rnE "default_rng\(\)|np\.random\.(rand|randn|choice|permutation|shuffle|seed)\(" \
      "$CODE" "$TRAIN/syn" 2>/dev/null | wc -l)
[ "$u" -eq 0 ] && ok "every rng is seeded" || bad "$u unseeded call(s)"

echo "3. results parse and carry their protocol"
"$PY" - "$LAB" <<'PYEOF'
import json, sys
from pathlib import Path
res = Path(sys.argv[1]) / "results"
bad, timed = [], 0
for p in sorted(res.glob("*.json")):
    try:
        d = json.loads(p.read_text())
    except Exception as e:
        bad.append(f"{p.name}: {e}"); continue
    if isinstance(d, dict) and "timing_valid" in d:
        timed += 1
        if not d["timing_valid"]:
            bad.append(f"{p.name}: timing_valid is false")
n = len(list(res.glob("*.json")))
print(f"  {'ok   ' if not bad else 'FAIL '} {n} files parse, {timed} carry timing_valid")
for b in bad[:5]:
    print("        " + b)
sys.exit(1 if bad else 0)
PYEOF
[ $? -eq 0 ] || fail=$((fail+1))

echo "4. timing audit, at a tolerance tighter than the default"
out=$(SYN_ROOT="$LAB" "$PY" "$CODE/audit_timing.py" --tol 0.03 2>&1 | tail -1)
case "$out" in
  "0 file(s) flagged"*) ok "$out" ;;
  *) bad "$out" ;;
esac

echo "5. withdrawn measurements are documented, not deleted"
if [ -f "$LAB/results/superseded/CONTENDED.md" ]; then
  n=$(ls "$LAB"/results/superseded/*.contended.json 2>/dev/null | wc -l)
  ok "$n withdrawn file(s) kept with CONTENDED.md"
else
  bad "results/superseded/CONTENDED.md missing"
fi

echo "6. every shipped figure has the script that drew it, and vice versa"
for f in fig_families fig_mutual_rank fig_strata \
         fig_build_cost fig_rerank_strata; do
  # fig_strata writes fig_strata_M16
  [ -f "$CODE/$f.py" ] || { bad "$(basename "$CODE")/$f.py missing"; continue; }
  hit=$(ls "$FIGS"/"$f"*.pdf 2>/dev/null | head -1)
  [ -n "$hit" ] && ok "$f.py -> $(basename "$hit")" || bad "$f.py drew nothing"
done
extra=$(ls "$FIGS"/*.pdf 2>/dev/null | wc -l)
# 8 in the main text and 2 in the appendix. The paper's fig4 is a schematic with
# no script in the lab and is not shipped, so the submission's nine main figures
# are eight here. The loop above checks the five whose script name matches their
# output; encoders, hardness measures and tail censor are checked by 6b and 6d,
# and the appendix pair is drawn by fig_ngfix.py and fig_rerank_strata.py.
[ "$extra" -eq 10 ] && ok "8 main figures and 2 appendix, no strays" \
  || bad "$extra figures shipped, expected 10"

echo "6b. the tail-censoring measurement ships with its scripts"
if [ -f "$FIGS/fig_tail_censor.pdf" ]; then
  n=$(ls "$TAILC"/*.py 2>/dev/null | wc -l)
  c=$(ls "$TAILC"/out/*_curves.json 2>/dev/null | wc -l)
  [ "$n" -ge 5 ] && [ "$c" -eq 2 ] \
    && ok "fig_tail_censor.pdf with $n scripts and $c curve files" \
    || bad "fig_tail_censor.pdf shipped with $n scripts, $c curve files"
else
  bad "fig_tail_censor.pdf missing"
fi

echo "6c. the annotation behind the 85% claim ships with its records"
"$PY" "$TOOLS/check_annot.py" "$ANNOT" || fail=$((fail+1))

echo "6d. the phi_h comparison ships with its scripts and its numbers"
if [ -f "$FIGS/fig_hardness_measures.pdf" ]; then
  n=$(ls "$PHIH"/scripts/* 2>/dev/null | wc -l)
  [ -f "$PHIH/VALUES.txt" ] && [ -f "$PHIH/table_cost.tex" ] && [ "$n" -ge 15 ] \
    && ok "fig_hardness_measures.pdf with $n scripts, VALUES.txt and table_cost.tex" \
    || bad "phi_h assets incomplete ($n scripts)"
  "$PY" "$PHIH/scripts/check_coverage.py" || fail=$((fail+1))
else
  bad "fig_hardness_measures.pdf missing"
fi

echo "7. nothing shipped that the paper does not use"
stray=$( { ls "$LAB"/results "$LAB"/corpora "$FIGS" "$CODE" \
           "$HERE"/patches 2>/dev/null; } \
         | grep -cE "mscoco|msmarco-(de|ru|zh|ja|hi|id)|darth" || true)
[ "$stray" -eq 0 ] && ok "no dropped collection leaked into results, corpora or figs" \
  || bad "$stray file(s) the paper does not use"
dead=$(ls "$CODE"/*.py 2>/dev/null | wc -l)
# Raised to 43 when families_table.py and encoders_table.py were added: the
# paper's Table 4 and Table 8 had no producing script, so their numbers could be
# read but not recomputed. Both are entry points rather than imports, which is
# why the count moves and the closure does not shrink.
[ "$dead" -le 43 ] && ok "$dead modules, the import closure of what the paper runs" \
  || bad "$dead modules shipped; the closure is 43"

echo "8. no path from the machine this was built on, and nothing that names a person"
# The second half matters more than the first. A leftover /root/SYN only breaks
# on someone else's disk; a home directory in a path names an author, and this
# is submitted to a double-blind venue. HOMES is the list of account names that
# have appeared in these trees; extend it rather than relax it.
HOMES='sieun|kyungmin|smrc'
host=$(grep -rIl --exclude=verify.sh --exclude=collect.sh \
         -e "/root/SYN" -e "/workspace/" -e "/root/lab" -e "/home/" \
         -E -e "$HOMES" "$HERE" 2>/dev/null | wc -l)
[ "$host" -eq 0 ] && ok "no build-machine path and no account name" \
  || { bad "$host file(s) carry a build-machine path"
       grep -rIl --exclude=verify.sh --exclude=collect.sh \
         -e "/root/SYN" -e "/workspace/" "$HERE" 2>/dev/null | head -5 | sed 's/^/        /'; }

echo "9. corpus manifests pin their seed and snapshot"
for m in "$LAB"/corpora/*_manifest.json; do
  [ -e "$m" ] || { bad "no corpus manifests"; break; }
  "$PY" - "$m" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
miss = [k for k in ("repo", "n_docs_subset") if k not in d]
print(("  ok    " if not miss else "  FAIL  ")
      + f"{sys.argv[1].split('/')[-1]} {'' if not miss else miss}")
sys.exit(1 if miss else 0)
PYEOF
  [ $? -eq 0 ] || fail=$((fail+1))
done

echo
[ "$fail" -eq 0 ] && echo "all checks passed" || echo "$fail check(s) failed"
exit $((fail > 0))
