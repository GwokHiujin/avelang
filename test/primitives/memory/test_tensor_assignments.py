#!/usr/bin/env python3
import unittest

import torch

import avelang
import avelang.language as S
import _avelang_bindings as _C
from avelang.compiler.code_generator import (
    _build_import_module,
    _collect_jit_dependencies,
    _get_function_def,
)


@avelang.jit
def kernel_scalar_assign(out: S.Tensor((8,), S.i32)):
    base = S.convert(7, S.i32)
    for i in S.range(8):
        out[i] = base + i


@avelang.jit
def kernel_vector_row_assign(
    out: S.Tensor((2, 4), S.u32),
    src: S.Tensor((2, 4), S.u32),
):
    row = src[0]
    row[2] = S.convert(99, S.u32)
    out[0] = row
    out[1] = src[1]


@avelang.jit
def kernel_memcpy(
    dst: S.Tensor((64, 4), S.u32),
    src: S.Tensor((64, 4), S.u32),
):
    thread_id = S.thread_id(0)
    dst[thread_id] = src[thread_id]


class TestTensorAssignments(unittest.TestCase):
    def test_scalar_assignment(self):
        out = torch.zeros((8,), dtype=torch.int32, device="cuda")
        expected = torch.arange(8, dtype=torch.int32, device="cuda") + 7

        kernel_scalar_assign[lambda: ((1, 1, 1), (1, 1, 1))](out)

        self.assertTrue(
            torch.equal(out.cpu(), expected.cpu()),
            f"Expected: {expected.tolist()}, Actual: {out.tolist()}",
        )

    def test_vector_row_assignment(self):
        src = torch.arange(2 * 4, dtype=torch.int32, device="cuda").view(2, 4)
        out = torch.zeros((2, 4), dtype=torch.int32, device="cuda")

        kernel_vector_row_assign[lambda: ((1, 1, 1), (1, 1, 1))](out, src)

        expected = src.clone()
        expected[0, 2] = 99

        self.assertTrue(
            torch.equal(out.cpu(), expected.cpu()),
            f"Expected: {expected.tolist()}, Actual: {out.tolist()}",
        )

    def test_vector_memcpy(self):
        src = torch.arange(64 * 4, dtype=torch.int32, device="cuda").view(64, 4)
        dst = torch.zeros((64, 4), dtype=torch.int32, device="cuda")

        kernel_memcpy[lambda: ((1, 1, 1), (64, 1, 1))](dst, src)

        self.assertTrue(
            torch.equal(dst.cpu(), src.cpu()),
            f"Expected: {src.tolist()}, Actual: {dst.tolist()}",
        )

    def test_vector_memcpy_mlir_uses_load_vec(self):
        jit_deps = _collect_jit_dependencies(kernel_memcpy)
        import_module = _build_import_module([kernel_memcpy, *jit_deps])

        generator = _C.MLIRGenerator()
        generator.generate_from_python_ast(import_module)

        for dep in jit_deps:
            dep_func = _get_function_def(dep.parse())
            generator.visit_function_def(dep_func, "[]", "jit")

        kernel_func = _get_function_def(kernel_memcpy.parse())
        generator.visit_function_def(kernel_func, "[]", "kernel")

        mlir = generator.get_mlir()
        self.assertIn("ave.memref.load_vec", mlir)


if __name__ == "__main__":
    unittest.main()
