#!/usr/bin/env python3
import json
import unittest

import torch

import avelang
import avelang.language as S
from tools import dump_assembly

GLOBAL_SIZE = 16
GLOBAL_OFFSET = S.constexpr(3)
GLOBAL_F32_EXACT = -0.5


@avelang.jit
def add_global(x: S.i32) -> S.i32:
    return x + GLOBAL_OFFSET


@avelang.jit
def kernel_global_constexpr(
    input_data: S.Tensor((GLOBAL_SIZE,), S.i32),
    output_data: S.Tensor((GLOBAL_SIZE,), S.i32),
):
    shared_buf = S.make_shared((GLOBAL_SIZE,), S.i32)
    tid = S.thread_id(0)
    shared_buf[tid] = add_global(input_data[tid])
    S.syncthreads()
    output_data[tid] = shared_buf[GLOBAL_SIZE - 1 - tid]


@avelang.jit
def kernel_global_exact_float_constant(
    output_data: S.Tensor((4,), S.f32),
):
    shared_buf = S.make_shared((4,), S.f32)
    tid = S.thread_id(0)
    shared_buf[tid] = GLOBAL_F32_EXACT
    S.syncthreads()
    output_data[tid] = shared_buf[tid]


@avelang.jit
def kernel_global_and_parameter_constexpr(
    output_data: S.Tensor((4,), S.i32), block_size: S.constexpr
):
    output_data[0] = GLOBAL_OFFSET + block_size


class TestGlobalConstexprInjection(unittest.TestCase):
    def test_assembly_dump_merges_global_and_parameter_constexprs(self):
        constexprs_json = json.dumps(
            [{"name": "block_size", "type": "i32", "value": 4}]
        )
        generator = dump_assembly._build_generator(
            kernel_global_and_parameter_constexpr, constexprs_json
        )
        self.assertIsNotNone(generator)

    def test_global_constexpr_injection(self):
        input_data = torch.arange(GLOBAL_SIZE, dtype=torch.int32, device="cuda")
        output_data = torch.zeros((GLOBAL_SIZE,), dtype=torch.int32, device="cuda")

        expected = (input_data + GLOBAL_OFFSET.value).flip(0)

        kernel_global_constexpr[lambda: ((1, 1, 1), (GLOBAL_SIZE, 1, 1))](input_data, output_data)

        actual = output_data.cpu()
        expected = expected.cpu()

        self.assertTrue(
            torch.equal(actual, expected),
            f"Expected: {expected.tolist()}, Actual: {actual.tolist()}",
        )

    def test_exact_float_constant_can_implicitly_demote(self):
        output_data = torch.zeros((4,), dtype=torch.float32, device="cuda")
        expected = torch.full((4,), GLOBAL_F32_EXACT, dtype=torch.float32, device="cuda")

        kernel_global_exact_float_constant[lambda: ((1, 1, 1), (4, 1, 1))](output_data)

        self.assertTrue(
            torch.equal(output_data.cpu(), expected.cpu()),
            f"Expected: {expected.tolist()}, Actual: {output_data.tolist()}",
        )


if __name__ == "__main__":
    unittest.main()
