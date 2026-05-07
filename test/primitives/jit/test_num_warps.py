#!/usr/bin/env python3
import unittest

import _avelang_bindings as _C
import avelang
import avelang.language as S
from avelang.testing import has_rocm

from avelang.compiler import code_generator as cg
from avelang.compiler.code_generator import (
    _build_import_module,
    _collect_jit_dependencies,
    _get_function_def,
)
from avelang.compiler.compiler import ASTSource


@avelang.jit
def kernel_with_num_warps(
    x: S.Tensor((1,), S.i32),
    BLOCK_SIZE: S.constexpr,
):
    tid = S.thread_id(0)
    if tid < BLOCK_SIZE:
        x[tid] = x[tid]


@unittest.skipUnless(
    has_rocm(),
    "Requires ROCm/HIP with an AMD GPU.",
)
class TestNumWarps(unittest.TestCase):
    def test_amdgpu_llvm_ir_overrides_flat_workgroup_size_when_num_warps_set(self):
        src = ASTSource(
            kernel_with_num_warps,
            {"x": "*i32", "BLOCK_SIZE": "constexpr"},
            constexprs={"BLOCK_SIZE": {"type": "i32", "value": 16}},
        )
        constexprs_json = cg._serialize_constexprs(src)

        generator = _C.MLIRGenerator()
        import_module = _build_import_module([kernel_with_num_warps, *_collect_jit_dependencies(kernel_with_num_warps)])
        generator.generate_from_python_ast(import_module)
        generator.visit_function_def(_get_function_def(kernel_with_num_warps.parse()), constexprs_json, "kernel")

        try:
            llvm_ir = generator.get_llvm_ir("amdgcn-amd-amdhsa", "gfx90a", 2, 3)
        except RuntimeError as exc:
            if "No backend available for target: amdgcn-amd-amdhsa" not in str(exc):
                raise
            self.skipTest("AMDGPU backend is not available.")
        self.assertIn('"amdgpu-flat-work-group-size"="1,192"', llvm_ir)


if __name__ == "__main__":
    unittest.main()
