#!/usr/bin/env bash
set -euo pipefail

LLVM_REPOSITORY="${LLVM_REPOSITORY:?LLVM_REPOSITORY is required}"
LLVM_REF="${LLVM_REF:?LLVM_REF is required}"
LLVM_TARGETS_TO_BUILD="${LLVM_TARGETS_TO_BUILD:?LLVM_TARGETS_TO_BUILD is required}"
LLVM_INSTALL_PREFIX="${LLVM_INSTALL_PREFIX:-/usr/local}"
LLVM_SOURCE_DIR="${LLVM_SOURCE_DIR:-/tmp/llvm-project}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-$LLVM_SOURCE_DIR/build}"
LLVM_DEFAULT_TARGET_TRIPLE="${LLVM_DEFAULT_TARGET_TRIPLE:-x86_64-unknown-linux-gnu}"
LLVM_C_COMPILER="${LLVM_C_COMPILER:-}"
LLVM_CXX_COMPILER="${LLVM_CXX_COMPILER:-}"

complete_marker="$LLVM_INSTALL_PREFIX/.avelang-llvm-${LLVM_REF//\//-}-${LLVM_TARGETS_TO_BUILD//[;\/]/-}.complete"

if [[ -f "$complete_marker" && -x "$LLVM_INSTALL_PREFIX/bin/mlir-opt" &&
      -f "$LLVM_INSTALL_PREFIX/include/mlir/Dialect/Affine/Passes.h" ]]; then
  "$LLVM_INSTALL_PREFIX/bin/llvm-config" --version
  "$LLVM_INSTALL_PREFIX/bin/mlir-opt" --version
  exit 0
fi

mkdir -p "$(dirname "$LLVM_SOURCE_DIR")" "$LLVM_INSTALL_PREFIX"

if [[ ! -d "$LLVM_SOURCE_DIR/.git" ]]; then
  rm -rf "$LLVM_SOURCE_DIR"
  git clone --depth 1 --branch "$LLVM_REF" "$LLVM_REPOSITORY" "$LLVM_SOURCE_DIR"
fi

rm -rf "$LLVM_BUILD_DIR"
mkdir -p "$LLVM_BUILD_DIR"
cd "$LLVM_BUILD_DIR"

cmake_env=()
if [[ -n "$LLVM_C_COMPILER" ]]; then
  cmake_env+=("CC=$LLVM_C_COMPILER")
fi
if [[ -n "$LLVM_CXX_COMPILER" ]]; then
  cmake_env+=("CXX=$LLVM_CXX_COMPILER")
fi

env "${cmake_env[@]}" cmake -G Ninja "$LLVM_SOURCE_DIR/llvm" \
  -DLLVM_ENABLE_PROJECTS="clang;lld;mlir" \
  -DLLVM_ENABLE_RUNTIMES="compiler-rt" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_ENABLE_EH=ON \
  -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_PREFIX" \
  -DCMAKE_INSTALL_RPATH="$LLVM_INSTALL_PREFIX/lib" \
  -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
  -DLLVM_TARGETS_TO_BUILD="$LLVM_TARGETS_TO_BUILD" \
  -DDEFAULT_TARGET_TRIPLE="$LLVM_DEFAULT_TARGET_TRIPLE" \
  -DLLVM_ENABLE_DUMP=ON

ninja "-j$(nproc)"
ninja install

affine_passes_header="$LLVM_INSTALL_PREFIX/include/mlir/Dialect/Affine/Passes.h"
affine_transforms_passes_header="$LLVM_INSTALL_PREFIX/include/mlir/Dialect/Affine/Transforms/Passes.h"
if [[ ! -f "$affine_passes_header" && -f "$affine_transforms_passes_header" ]]; then
  mkdir -p "$(dirname "$affine_passes_header")"
  printf '%s\n' \
    '#ifndef MLIR_DIALECT_AFFINE_PASSES_H' \
    '#define MLIR_DIALECT_AFFINE_PASSES_H' \
    '#include "mlir/Dialect/Affine/Transforms/Passes.h"' \
    '#endif // MLIR_DIALECT_AFFINE_PASSES_H' \
    > "$affine_passes_header"
fi

"$LLVM_INSTALL_PREFIX/bin/llvm-config" --version
"$LLVM_INSTALL_PREFIX/bin/mlir-opt" --version
test -f "$affine_passes_header"
touch "$complete_marker"
