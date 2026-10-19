#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

# shellcheck source=image.env
source "${script_dir}/image.env"

image="${1:-${AVELANG_CI_IMAGE}}"
runtime_args=()
if [[ -n "${AVE_OCI_RUNTIME:-}" ]]; then
  runtime_args+=(--runtime "${AVE_OCI_RUNTIME}")
fi

podman run --rm \
  "${runtime_args[@]}" \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add keep-groups \
  --ipc=host \
  --mount type=tmpfs,destination=/dev/shm,tmpfs-size=34359738368 \
  --ulimit memlock=-1:-1 \
  --volume "${repo_root}:/src/avelang:ro" \
  "${image}" \
  bash -lc '
    set -euo pipefail

    test "$AMDGPU_TARGETS" = gfx950
    test "$PYTORCH_ROCM_ARCH" = gfx950
    test "$(llvm-config --version)" = 23.0.0git
    test "$(llvm-config --has-rtti)" = YES
    test "$(llvm-config --targets-built)" = "X86 AMDGPU"
    test "$(/opt/actions-runner/bin/Runner.Listener --version)" = 2.337.0
    rocminfo | grep -m1 gfx950
    python -c "import torch; assert torch.version.hip; assert torch.cuda.is_available(); print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))"

    mkdir -p /tmp/avelang
    tar -C /src/avelang \
      --exclude=.git \
      --exclude=build \
      --exclude=.pytest_cache \
      --exclude='*/__pycache__' \
      --exclude='*.pyc' \
      --exclude=python/avelang_kernels/KDA \
      --exclude="python/*.so" \
      --exclude="python/*.egg-info" \
      -cf - . | tar -C /tmp/avelang -xf -

    cd /tmp/avelang
    rocm_prefix="$(hipconfig --rocmpath)"
    cmake --fresh -S . -B build/ci -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DAVE_LANG_BACKEND=rocm \
      -DWITH_PYTHON=ON \
      -DCMAKE_C_COMPILER=/opt/llvm23/bin/clang \
      -DCMAKE_CXX_COMPILER=/opt/llvm23/bin/clang++ \
      -DCMAKE_PREFIX_PATH="/opt/llvm23;${rocm_prefix}" \
      -DCMAKE_EXE_LINKER_FLAGS=-Wl,--exclude-libs,ALL \
      -DCMAKE_MODULE_LINKER_FLAGS=-Wl,--exclude-libs,ALL \
      -DPython3_ROOT_DIR=/opt/venv \
      -DPython3_EXECUTABLE=/opt/venv/bin/python \
      -DAVE_LANG_PYTHON_LIBRARY_OUTPUT_DIRECTORY=/tmp/avelang/python
    ninja -C build/ci -j"$(nproc)"
    ctest --test-dir build/ci --output-on-failure --parallel "$(nproc)"
    python setup.py egg_info --egg-base python
    PYTHONPATH=/tmp/avelang/python python -m pytest -q -ra test/
  '
