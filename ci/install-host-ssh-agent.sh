#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
unit_name=host-ssh-agent.service
unit_dir="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"
bin_dir="${HOME}/.local/bin"
private_key="${HOME}/.ssh/id_ed25519"

if [[ ! -r "${private_key}" ]]; then
  echo "Host SSH key is unavailable at ${private_key}." >&2
  exit 1
fi

key_mode="$(stat -c '%a' "${private_key}")"
if [[ "${key_mode}" != 600 ]]; then
  echo "Refusing to load ${private_key}: expected mode 600, found ${key_mode}." >&2
  exit 1
fi

install -D -m 0644 "${script_dir}/${unit_name}" "${unit_dir}/${unit_name}"
install -D -m 0755 "${script_dir}/podman-with-host-ssh-agent" \
  "${bin_dir}/podman-with-host-ssh-agent"

systemctl --user daemon-reload
systemctl --user enable "${unit_name}"
systemctl --user restart "${unit_name}"

agent_socket="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/host-ssh-agent.sock"
SSH_AUTH_SOCK="${agent_socket}" ssh-add -l

echo "Installed ${bin_dir}/podman-with-host-ssh-agent"
echo "The private key remains on the host; opted-in containers receive only ${agent_socket}."
