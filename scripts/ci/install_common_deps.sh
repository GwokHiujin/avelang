#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  SUDO=sudo
else
  SUDO=
fi

$SUDO apt-get update
$SUDO apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  cmake \
  curl \
  g++ \
  gcc \
  git \
  ninja-build \
  python3-dev \
  python3-pip \
  python3-venv \
  zlib1g-dev \
  zstd

cmake --version
ninja --version
python3 --version
