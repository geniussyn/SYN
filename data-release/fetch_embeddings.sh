#!/usr/bin/env bash
# Fetch the eight embedding matrices and their four manifests into emb/.
#
# Each file is verified against scripts/embeddings.manifest. A file that is
# already present and already correct is skipped, so an interrupted run is
# resumed by running this again. Nothing is written in place: a download lands
# in <name>.part and is renamed only after its checksum matches, so a failure
# can never leave a corrupt .npy where a figure script will read it.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
DEST="${1:-$REPO/emb}"
MANIFEST="$HERE/embeddings.manifest"
IDS="$HERE/embeddings.ids"

[ -f "$MANIFEST" ] || { echo "missing $MANIFEST"; exit 2; }
[ -f "$IDS" ] || {
  cat <<'MSG'
scripts/embeddings.ids is not present, so there is nowhere to fetch from.

That file maps each filename to its file id on the host and is written when the
release is published. Until then, either obtain it with the paper's data link or
regenerate the embeddings locally -- see docs/DATA.md, "Regenerating them
instead". The regeneration path needs no download and no id.
MSG
  exit 2
}

mkdir -p "$DEST" || exit 2
command -v curl >/dev/null || { echo "curl is required"; exit 2; }

# name -> expected sha256
declare -A WANT
while read -r sum name; do
  case "$sum" in ''|'#'*) continue ;; esac
  WANT["$name"]="$sum"
done < "$MANIFEST"

# name -> file id
declare -A ID
while read -r name fid; do
  case "$name" in ''|'#'*) continue ;; esac
  ID["$name"]="$fid"
done < "$IDS"

ok=0; skipped=0; failed=(); stale=()

# Google Drive serves anything over ~100 MB behind a confirmation page. The
# first request collects the cookie and the token; the second redeems them.
# A quota refusal and an expired token both come back as that same HTML page,
# which is why the size and the checksum are both checked afterwards.
fetch() {
  local fid="$1" out="$2" jar
  jar="$(mktemp)"
  local url="https://drive.usercontent.google.com/download?id=${fid}&export=download"
  local token
  token="$(curl -sL -c "$jar" "$url" \
           | grep -o 'name="confirm" value="[^"]*"' | head -1 \
           | sed 's/.*value="\([^"]*\)".*/\1/')"
  if [ -n "$token" ]; then
    curl -L -b "$jar" -o "$out" --fail --retry 3 --retry-delay 5 \
         --progress-bar "${url}&confirm=${token}"
  else
    curl -L -b "$jar" -o "$out" --fail --retry 3 --retry-delay 5 \
         --progress-bar "$url"
  fi
  local rc=$?
  rm -f "$jar"
  return $rc
}

for name in $(awk '!/^#/ && NF==2 {print $2}' "$MANIFEST"); do
  want="${WANT[$name]}"
  target="$DEST/$name"

  if [ -f "$target" ]; then
    have="$(sha256sum "$target" | cut -d' ' -f1)"
    if [ "$have" = "$want" ]; then
      printf '  ok       %s\n' "$name"; skipped=$((skipped+1)); continue
    fi
    printf '  stale    %s (checksum differs, refetching)\n' "$name"
    stale+=("$name")
  fi

  fid="${ID[$name]:-}"
  if [ -z "$fid" ]; then
    printf '  NO ID    %s\n' "$name"; failed+=("$name"); continue
  fi

  printf '  fetch    %s\n' "$name"
  if ! fetch "$fid" "$target.part"; then
    printf '  FAILED   %s (transfer error)\n' "$name"
    rm -f "$target.part"; failed+=("$name"); continue
  fi

  # An HTML error page is a few KB; the smallest real file here is 13 MB.
  sz=$(stat -c%s "$target.part" 2>/dev/null || echo 0)
  if [ "$sz" -lt 1000000 ] && head -c 512 "$target.part" | grep -qi '<html'; then
    printf '  FAILED   %s (got the confirmation or quota page, not the file)\n' "$name"
    rm -f "$target.part"; failed+=("$name"); continue
  fi

  have="$(sha256sum "$target.part" | cut -d' ' -f1)"
  if [ "$have" != "$want" ]; then
    printf '  FAILED   %s (checksum mismatch, %d bytes)\n' "$name" "$sz"
    printf '           want %s\n           got  %s\n' "$want" "$have"
    rm -f "$target.part"; failed+=("$name"); continue
  fi

  mv "$target.part" "$target"
  # replaced, so it is no longer a wrong file sitting on disk
  for i in "${!stale[@]}"; do
    [ "${stale[$i]}" = "$name" ] && unset 'stale[$i]'
  done
  printf '  ok       %s\n' "$name"; ok=$((ok+1))
done

echo
printf '%d fetched, %d already correct, %d failed -> %s\n' \
       "$ok" "$skipped" "${#failed[@]}" "$DEST"

if [ "${#failed[@]}" -gt 0 ]; then
  printf '\nnot obtained:\n'
  for n in "${failed[@]}"; do printf '  %s\n' "$n"; done
  cat <<'MSG'

A quota refusal clears on its own, usually within a day; re-running this script
resumes from what is already correct. To take the manual path, download the
files in a browser into the destination directory and check them with

    cd <destination> && sha256sum -c <repo>/scripts/embeddings.manifest

Or skip the download entirely and regenerate -- see docs/DATA.md.
MSG
  if [ "${#stale[@]}" -gt 0 ]; then
    printf '\nSTILL WRONG ON DISK -- these were already there, do not match the\n'
    printf 'manifest, and could not be replaced. Do not measure against them:\n'
    for n in "${stale[@]}"; do printf '  %s/%s\n' "$DEST" "$n"; done
    printf 'Delete them, or re-run once the fetch works.\n'
  fi
  exit 1
fi

if [ "$DEST" != "$REPO/emb" ]; then
  printf '\nexport SYN_EMB_DIR=%s\n' "$DEST"
fi
