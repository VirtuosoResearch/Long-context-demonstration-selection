# retag_ghcr_to_harness.sh
#!/usr/bin/env bash
python ./notebooks/fetch_offline_image.py

set -euo pipefail
ARCH="x86_64"
SRC_PREFIX="ghcr.io/epoch-research/swe-bench.eval.${ARCH}"
DST1="swe-bench/eval/${ARCH}"
DST2="sweb.eval.${ARCH}"
DST3="swebench/sweb.eval.${ARCH}"

to_1776 () {
  local iid="$1"  # e.g., django__django-11001
  if [[ "$iid" =~ ^([A-Za-z0-9]+)__(.+)$ ]]; then
    echo "${BASH_REMATCH[1]}_1776_${BASH_REMATCH[2]}"
  else
    echo ""
  fi
}

mapfile -t REFS < <(docker images --format '{{.Repository}}:{{.Tag}}' \
  | grep "^${SRC_PREFIX}\." | grep ':latest$' || true)

for REF in "${REFS[@]}"; do
  iid="${REF#${SRC_PREFIX}.}"; iid="${iid%:latest}"

  docker tag "$REF" "${DST1}/${iid}:latest" || true
  docker tag "$REF" "${DST2}.${iid}:latest" || true
  docker tag "$REF" "${DST3}.${iid}:latest" || true

  v1776="$(to_1776 "$iid")"
  if [[ -n "$v1776" ]]; then
    docker tag "$REF" "${DST3}.${v1776}:latest" || true
    docker tag "$REF" "${DST2}.${v1776}:latest" || true
  fi
done
echo "done"

# chmod +x retag_ghcr_to_harness.sh
# ./retag_ghcr_to_harness.sh
# docker images --filter reference='swebench/sweb.eval.x86_64.*:latest'