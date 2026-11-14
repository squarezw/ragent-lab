#!/usr/bin/env bash
# Run from an extracted release directory. Images are immutable commit SHA tags.
set -Eeuo pipefail
release_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base_dir="${LAB_DEPLOY_ROOT:-/opt/rag-lab}"
revision="${1:?usage: deploy.sh COMMIT_SHA}"
[[ "$revision" =~ ^[a-f0-9]{40}$ ]] || { echo 'Invalid revision'; exit 1; }
exec 9>"$base_dir/.deploy.lock"
flock -w 600 9 || { echo 'Another deployment is running'; exit 1; }
previous="$(readlink -f "$base_dir/current" 2>/dev/null || true)"
legacy_running="$(docker inspect -f '{{.State.Running}}' rag-lab 2>/dev/null || true)"
[[ ! -f "$base_dir/.env" ]] || install -m 600 "$base_dir/.env" "$release_dir/.env"
compose() { docker compose --project-name ragent-lab --env-file "$release_dir/release.env" -f "$release_dir/docker/docker-compose.yml" "$@"; }
# Deterministic ownership/permissions for the read-only samples bind mount.
install -d -m 0755 "${LAB_VISION_SAMPLES:-$base_dir/vision-samples}"
# The tuned prompts live only on the server (see vision/prompts.py); never created empty here
# with content, just made sure the mount source exists.
install -d -m 0755 "${LAB_VISION_PROMPTS:-$base_dir/vision-prompts}"
# Pull before stopping the live legacy service.
if [[ "${LAB_SKIP_PULL:-false}" != true ]]; then compose pull; fi
rollback() {
    echo 'Deployment failed; restoring previous service.'
    compose down --remove-orphans || true
    if [[ -n "$previous" && -f "$previous/release.env" ]]; then
        docker compose --project-name ragent-lab --env-file "$previous/release.env" -f "$previous/docker/docker-compose.yml" up -d --wait --wait-timeout 180 || true
    elif [[ "$legacy_running" == true ]]; then
        docker start rag-lab || true
    fi
}
trap rollback ERR
if [[ "$legacy_running" == true ]]; then docker stop rag-lab; fi
compose up -d --no-build --remove-orphans --wait --wait-timeout 240
# Verify the real public-path proxy and exact release before promoting it.
for attempt in $(seq 1 20); do
    if curl -fsS http://127.0.0.1:8501/api/health | grep -q "$revision" && curl -fsS http://127.0.0.1:8501/ | grep -q 'Ragent Lab' && curl -fsS --resolve lab.ragents.net:443:127.0.0.1 https://lab.ragents.net/api/health | grep -q "$revision"; then
        ln -sfn "$release_dir" "$base_dir/current"
        trap - ERR
        echo "Deployed $revision"
        exit 0
    fi
    sleep 3
done
false
