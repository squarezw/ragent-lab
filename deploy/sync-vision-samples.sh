#!/usr/bin/env bash
# Usage: deploy/sync-vision-samples.sh <local sample dir>
# Each sample is a subfolder holding rubric.json plus its image.
set -euo pipefail
dir="${1:?usage: $0 <local sample dir>}"
[ -d "$dir" ] || { echo "not a directory: $dir" >&2; exit 1; }
if ! compgen -G "$dir/*/rubric.json" > /dev/null; then
  echo "no */rubric.json found in $dir" >&2
  exit 1
fi
HOST=root@209.209.40.180
rsync -av --delete --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r --no-owner --no-group "$dir"/ "$HOST":/opt/rag-lab/vision-samples/
ssh "$HOST" 'cd /opt/rag-lab/current && docker compose --project-name ragent-lab --env-file release.env -f docker/docker-compose.yml restart api'
