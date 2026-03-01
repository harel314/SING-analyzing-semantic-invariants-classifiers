#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

docker compose -f "${PROJECT_ROOT}/docker/docker-compose.gpu.yml" pull
docker compose -f "${PROJECT_ROOT}/docker/docker-compose.gpu.yml" up -d
docker compose -f "${PROJECT_ROOT}/docker/docker-compose.gpu.yml" ps
