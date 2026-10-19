#!/usr/bin/env bash
set -euo pipefail

runner_dir="${RUNNER_STATE_DIR:-/workspace/ave_action_runner}"
runner_seed_dir=/opt/actions-runner

if [[ ! -f "${runner_dir}/.runner" || ! -f "${runner_dir}/.credentials" ]]; then
  echo "Runner registration is missing from ${runner_dir}." >&2
  echo "Register the runner once before starting the service." >&2
  exit 1
fi

# Refresh the executable payload from the pinned image while preserving the
# registration, credentials, diagnostics, and work directory on the host.
cp -a "${runner_seed_dir}/." "${runner_dir}/"
chmod 600 "${runner_dir}/.credentials" "${runner_dir}/.credentials_rsaparams"

cd "${runner_dir}"
exec ./run.sh
