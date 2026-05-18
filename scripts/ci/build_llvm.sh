#!/usr/bin/env bash
set -euo pipefail

LLVM_REPOSITORY="${LLVM_REPOSITORY:?LLVM_REPOSITORY is required}"
LLVM_REF="${LLVM_REF:?LLVM_REF is required}"
LLVM_INSTALL_PREFIX="${LLVM_INSTALL_PREFIX:-/opt/llvm-ci}"
LLVM_SOURCE_DIR="${LLVM_SOURCE_DIR:-/tmp/llvm-project}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-/tmp/llvm-build}"
LLVM_PARALLEL_LINK_JOBS="${LLVM_PARALLEL_LINK_JOBS:-2}"
LLVM_TARGETS_TO_BUILD="${LLVM_TARGETS_TO_BUILD:-all}"

complete_marker="$LLVM_INSTALL_PREFIX/.avelang-llvm-${LLVM_REF//\//-}-rtti-eh-all-targets.complete"

if [[ -f "$complete_marker" && -x "$LLVM_INSTALL_PREFIX/bin/mlir-opt" ]]; then
  "$LLVM_INSTALL_PREFIX/bin/llvm-config" --version
  "$LLVM_INSTALL_PREFIX/bin/mlir-opt" --version
  exit 0
fi

mkdir -p "$(dirname "$LLVM_SOURCE_DIR")" "$(dirname "$LLVM_BUILD_DIR")" "$LLVM_INSTALL_PREFIX"

if [[ ! -d "$LLVM_SOURCE_DIR/.git" ]]; then
  rm -rf "$LLVM_SOURCE_DIR"
  git clone --depth 1 --branch "$LLVM_REF" "$LLVM_REPOSITORY" "$LLVM_SOURCE_DIR"
fi

cmake -S "$LLVM_SOURCE_DIR/llvm" -B "$LLVM_BUILD_DIR" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_PREFIX" \
  -DLLVM_ENABLE_PROJECTS="clang;lld;mlir" \
  -DLLVM_ENABLE_RUNTIMES="compiler-rt" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_ENABLE_EH=ON \
  -DLLVM_TARGETS_TO_BUILD="$LLVM_TARGETS_TO_BUILD" \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_PARALLEL_LINK_JOBS="$LLVM_PARALLEL_LINK_JOBS" \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_CXX_FLAGS="-Wno-error=stringop-overflow -Wno-error=deprecated-declarations" \
  -DCMAKE_C_FLAGS="-Wno-error=stringop-overflow -Wno-error=deprecated-declarations" \
  -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON

cmake --build "$LLVM_BUILD_DIR" --target install

"$LLVM_INSTALL_PREFIX/bin/llvm-config" --version
"$LLVM_INSTALL_PREFIX/bin/mlir-opt" --version
touch "$complete_marker"
