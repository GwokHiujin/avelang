#!/usr/bin/env bash
set -euo pipefail

backend="${1:?usage: run_compiler_ci.sh <cuda|rocm>}"

LLVM_INSTALL_PREFIX="${LLVM_INSTALL_PREFIX:-/usr/local}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

case "$backend" in
  cuda)
    backend_cmake="cuda"
    prefix_path="$LLVM_INSTALL_PREFIX"
    ;;
  rocm)
    backend_cmake="rocm"
    ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
    export ROCM_PATH
    export PATH="$ROCM_PATH/bin:$PATH"
    prefix_path="$LLVM_INSTALL_PREFIX;$ROCM_PATH"
    ;;
  *)
    echo "Unsupported backend '$backend'; expected 'cuda' or 'rocm'." >&2
    exit 2
    ;;
esac

if [[ ! -x "$LLVM_INSTALL_PREFIX/bin/clang" ]]; then
  echo "Missing clang under LLVM_INSTALL_PREFIX=$LLVM_INSTALL_PREFIX" >&2
  exit 1
fi

export PATH="$LLVM_INSTALL_PREFIX/bin:$PATH"

common_cmake_args=(
  "-DAVE_LANG_BACKEND=$backend_cmake"
  "-DWITH_PYTHON=ON"
  "-DCMAKE_BUILD_TYPE=Release"
  "-DCMAKE_C_COMPILER=$LLVM_INSTALL_PREFIX/bin/clang"
  "-DCMAKE_CXX_COMPILER=$LLVM_INSTALL_PREFIX/bin/clang++"
  "-DCMAKE_PREFIX_PATH=$prefix_path"
)

$PYTHON_BIN -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip uv
uv pip install numpy pybind11 pytest ruff isort
pybind11_cmake_dir="$(python -m pybind11 --cmakedir)"
common_cmake_args+=("-Dpybind11_DIR=$pybind11_cmake_dir")

CMAKE_ARGS="${common_cmake_args[*]}" uv pip install -e . --no-deps

python - <<'PY'
import _avelang_bindings
import avelang

print("_avelang_bindings:", _avelang_bindings.__file__)
print("avelang:", avelang.__file__)
PY

rm -rf build
cmake -S . -B build -GNinja \
  "${common_cmake_args[@]}" \
  "-DAVE_LANG_PYTHON_LIBRARY_OUTPUT_DIRECTORY=$PWD/python"

ninja -C build
ctest --test-dir build --output-on-failure -E '^(amdgpu_codegen_test|nvvm_codegen_test)$'
