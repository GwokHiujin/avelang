#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

# shellcheck source=image.env
source "${script_dir}/image.env"

image="${1:-${AVELANG_CI_IMAGE}}"
jobs="${LLVM_BUILD_JOBS:-32}"

podman build \
  --file "${script_dir}/Containerfile" \
  --tag "${image}" \
  --build-arg "LLVM_BUILD_JOBS=${jobs}" \
  "${repo_root}"

podman image inspect "${image}" --format '{{.Id}} {{.Digest}}'
