#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
unit_name=avelang-ci-runner.service
unit_dir="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"
container_name=ave-kda-mi355

if podman container exists "${container_name}" \
    && podman exec "${container_name}" pgrep -x Runner.Worker >/dev/null 2>&1; then
  echo "A GitHub Actions job is active; refusing to restart ${container_name}." >&2
  exit 1
fi

loginctl enable-linger "${USER}"
if [[ "$(loginctl show-user "${USER}" --property=Linger --value)" != yes ]]; then
  echo "User lingering is required for boot and logout persistence." >&2
  exit 1
fi

install -D -m 0644 "${script_dir}/${unit_name}" "${unit_dir}/${unit_name}"
systemctl --user daemon-reload
systemctl --user enable --now "${unit_name}"
