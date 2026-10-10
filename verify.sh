#!/bin/bash
# Check the claims ARTIFACT.md makes. Reads; never measures.
#
#   bash verify.sh              against the assembled artifact
#   LAB=/root/SYN bash verify.sh   against the working tree instead
#
# The lab keeps code/ and its side studies under one working root; the
# repository lays them out as measure/ and studies/. Both are
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
  CODE="$LAB/measure"; FIGS="$LAB/figures"; PHIH="$LAB/studies/hardness"
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

echo "3. the measurement code is whole"
# The artifact ships what derives a result, not the result. So the check is
# that the deriving code is complete: every module compiles, and the import
# closure closes -- a module that reaches for something left behind would fail
# only when a reviewer ran it.
# compileall writes __pycache__ beside the sources, and a .pyc carries the
# absolute path of the machine that wrote it -- which check 8 then reports as a
# build-machine path. Whatever this check creates, it removes; so does check 1,
# which imports the training package.
if out=$("$PY" -m compileall -q "$CODE" "$TRAIN/src" 2>&1); then
  ok "$(ls "$CODE"/*.py | wc -l) modules in $(basename "$CODE")/ compile, with src/"
else
  bad "a module does not compile"; echo "$out" | tail -4
fi
find "$LAB" "$TRAIN" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
miss=$("$PY" - "$CODE" <<'PYEOF'
import ast, pathlib, sys
d = pathlib.Path(sys.argv[1])
have = {f.stem for f in d.glob("*.py")} | {"syn"}
missing = set()
for f in d.glob("*.py"):
    for n in ast.walk(ast.parse(f.read_text())):
        if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            m = n.module.split(".")[0]
        elif isinstance(n, ast.Import):
            for a in n.names:
                m = a.name.split(".")[0]
                if (d / f"{m}.py").exists() or m in have:
                    continue
            continue
        else:
            continue
        if m in have:
            continue
        try:
            __import__(m)
        except Exception:
            missing.add(f"{f.stem} -> {m}")
print("\n".join(sorted(missing)))
PYEOF
)
[ -z "$miss" ] && ok "every import resolves, inside the tree or in the environment" \
  || { bad "unresolved imports"; echo "$miss" | head -5 | sed 's/^/        /'; }

echo "4. nothing that draws came along"
drew=$(ls "$CODE"/fig_*.py "$CODE"/*_table.py "$CODE"/housestyle.py \
          "$CODE"/style.py "$CODE"/gnuplot_all.py "$CODE"/export_gnuplot.py \
          2>/dev/null | wc -l)
figdirs=$(ls -d "$LAB"/figures "$LAB"/gnuplot "$LAB"/results 2>/dev/null | wc -l)
if [ "$drew" -eq 0 ] && [ "$figdirs" -eq 0 ]; then
  ok "no figure module, no rendered figure, no measured result"
else
  bad "$drew drawing module(s) and $figdirs output directory(ies) shipped"
fi

echo "5. the experiments that derive the paper's numbers are here"
need="run_all queue ablation encoders crosslingual ngfix fever_sweeps"
got=0; for w in $need; do [ -f "$LAB/experiments/$w.sh" ] && got=$((got+1)); done
[ "$got" -eq 7 ] && ok "$got lanes, and $(ls "$LAB"/experiments/*.sh | wc -l) in all" \
  || bad "$got of 7 named lanes present"

echo "6. the three side studies keep the evidence no code can recompute"
# These three are not re-runnable here and are not meant to be: SpaceV's base
# is 10 GB of int8, the phi_h comparison ran on other corpora, and the
# annotation is a human judgement. For them the recorded output IS the thing.
n=$(ls "$ANNOT"/*.jsonl 2>/dev/null | wc -l)
[ "$n" -gt 0 ] && [ -f "$ANNOT/GUIDELINES.md" ] \
  && ok "annotation: $n record file(s) with the guidelines they were made under" \
  || bad "the 85% claim ships without its records"
[ -f "$PHIH/VALUES.txt" ] && [ -d "$PHIH/scripts" ] \
  && ok "phi_h: VALUES.txt with $(ls "$PHIH"/scripts | wc -l) scripts" \
  || bad "the phi_h comparison ships without its numbers or its scripts"
[ -f "$TAILC/RESULTS.md" ] && [ -n "$(ls "$TAILC"/out/*.json 2>/dev/null)" ] \
  && ok "tail censoring: curves and the ground-truth audit" \
  || bad "the tail-censoring measurement ships without its curves"

echo "7. the training's product ships, and can regenerate the embeddings"
na=$(ls -d "$LAB"/adapters/*/ 2>/dev/null | wc -l)
nw=$(ls "$LAB"/adapters/*/adapter_model.safetensors 2>/dev/null | wc -l)
if [ "$na" -ge 7 ] && [ "$nw" -eq "$na" ]; then
  ok "$na adapter(s), each with weights ($(du -sh "$LAB/adapters" | cut -f1))"
else
  bad "$na adapter dir(s) but $nw weight file(s)"
fi

echo "8. no path from the machine this was built on, and nothing that names a person"
# The second half matters more than the first. A leftover /root/SYN only breaks
# on someone else's disk; a home directory in a path names an author, and this
# is submitted to a double-blind venue.
#
# The names themselves are NOT written here. They used to be, on the reasoning
# that keeping them out of ANONYMITY.md stopped that note becoming the leak it
# warns about -- which missed that this file ships too, so the list was
# published instead. What ships now is the shape of the leak, not an instance
# of it: absolute build paths, any home directory at all, and the form of a
# hosting URL. The author adds the literal names from outside the tree:
#
#     SYN_NAMES='name1|name2' bash verify.sh
#
# A reviewer who does not set it still gets every structural check; only the
# author can run the one that needs to know what to look for.
#
# .git/ is excluded on purpose: it is not distributed -- a clone builds its own
# -- and the remote URL that lives in its config is how the author pushes. The
# check is about the working tree, which is what a reader receives.
PATTERNS=(-e "/root/SYN" -e "/workspace/" -e "/root/lab" -e "/home/"
          -E -e "github\.com/[A-Za-z0-9_-]+/(SYN|syn)[A-Za-z0-9_-]*")
if [ -n "${SYN_NAMES:-}" ]; then
  PATTERNS+=(-E -e "$SYN_NAMES")
  scope="paths, hosting URLs and the supplied names"
else
  scope="paths and hosting URLs (set SYN_NAMES to add the author list)"
fi
host=$(grep -rIl --exclude-dir=.git --exclude=verify.sh --exclude=collect.sh \
         "${PATTERNS[@]}" "$HERE" 2>/dev/null | wc -l)
[ "$host" -eq 0 ] && ok "nothing matching $scope" \
  || { bad "$host file(s) carry a build-machine path, a hosting URL or a name"
       grep -rIl --exclude-dir=.git --exclude=verify.sh --exclude=collect.sh \
         "${PATTERNS[@]}" "$HERE" 2>/dev/null \
         | sed "s|$HERE/||" | head -5 | sed 's/^/        /'; }

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

echo "10. the embeddings the paper measured are addressable and checkable"
man="$LAB/experiments/embeddings.manifest"
if [ ! -f "$man" ]; then
  bad "no experiments/embeddings.manifest -- nothing pins the bytes the paper used"
else
  nsum=$(awk '!/^#/ && NF==2' "$man" | wc -l)
  # four arms: docs + queries + manifest
  if [ "$nsum" -eq 12 ]; then
    ok "$nsum checksums (msmarco-1m and fever, vanilla and syn)"
  else
    bad "$nsum checksums, expected 12"
  fi
  # A published release must carry the ids; an anonymous one must not leak a
  # host account. Both states are legal, so this only reports which one it is.
  if [ -f "$LAB/experiments/embeddings.ids" ]; then
    nid=$(awk '!/^#/ && NF==2' "$LAB/experiments/embeddings.ids" | wc -l)
    [ "$nid" -eq "$nsum" ] && ok "embeddings.ids covers all $nid files" \
      || bad "embeddings.ids has $nid of $nsum ids -- fetch will refuse the rest"
  else
    ok "no embeddings.ids yet; fetch refuses and points at the rebuild path"
  fi
  # The regeneration path is the fallback the data note promises, so the two
  # things it needs have to be present whatever the host is doing.
  [ -f "$CODE/encode_cache.py" ] && [ -d "$LAB/adapters/syn" ] \
    && ok "the rebuild path is present (encode_cache.py + adapters/syn)" \
    || bad "docs/DATA.md promises a rebuild path that does not ship"
fi

echo
[ "$fail" -eq 0 ] && echo "all checks passed" || echo "$fail check(s) failed"
exit $((fail > 0))
