#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/image.env"

wrapper="${HOME}/.local/bin/podman-with-host-ssh-agent"
if [[ ! -x "${wrapper}" ]]; then
  echo "Install the host agent first: ${script_dir}/install-host-ssh-agent.sh" >&2
  exit 1
fi

exec "${wrapper}" \
  --rm \
  --name avelang-dev-mi355 \
  --runtime="${HOME}/.local/crun-package/crun-local" \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=keep-groups \
  --ipc=host \
  --mount=type=tmpfs,destination=/dev/shm,tmpfs-size=34359738368 \
  --ulimit=memlock=-1:-1 \
  --volume="${HOME}/dev:/workspace" \
  --workdir=/workspace/avelang \
  --env=AMDGPU_TARGETS=gfx950 \
  --env=PYTORCH_ROCM_ARCH=gfx950 \
  --interactive \
  --tty \
  "${AVELANG_CI_IMAGE}" \
  bash
