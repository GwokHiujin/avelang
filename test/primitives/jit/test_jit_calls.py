#!/usr/bin/env python3
import unittest

import torch
import avelang
import avelang.language as S


def helper(x):
    return x + 1


@avelang.jit
def add_values_jit(
    a: S.f32,
    b: S.f32,
) -> S.f32:
    return a + b


@avelang.jit
def kernel_add(
    a: S.Tensor((64,), S.f32),
    b: S.Tensor((64,), S.f32),
    c: S.Tensor((64,), S.f32),
):
    tid = S.thread_id(0)
    c[tid] = add_values_jit(a[tid], b[tid])


@avelang.jit
def kernel_add_nested(
    a: S.Tensor((64,), S.f32),
    b: S.Tensor((64,), S.f32),
    c: S.Tensor((64,), S.f32),
):
    def add_values_inner(x: S.f32, y: S.f32) -> S.f32:
        return x + y

    tid = S.thread_id(0)
    c[tid] = add_values_inner(a[tid], b[tid])


class TestJitCalls(unittest.TestCase):
    """Test that avelang.jit functions can be called from avelang.jit kernels."""

    def test_jit_function(self):
        WARP_SIZE = 64
        A = torch.randn((WARP_SIZE,), dtype=torch.float32, device="cuda")
        B = torch.randn((WARP_SIZE,), dtype=torch.float32, device="cuda")
        C = torch.zeros((WARP_SIZE,), dtype=torch.float32, device="cuda")

        expected = (A + B).to(dtype=torch.float32, device="cpu")
        kernel_add[lambda: ((1, 1, 1), (64, 1, 1))](A, B, C)

        actual = C.cpu()

        self.assertTrue(
            torch.allclose(actual, expected),
            f"Expected: {expected.tolist()}, Actual: {actual.tolist()}",
        )

    def test_nested_function(self):
        WARP_SIZE = 64
        A = torch.randn((WARP_SIZE,), dtype=torch.float32, device="cuda")
        B = torch.randn((WARP_SIZE,), dtype=torch.float32, device="cuda")
        C = torch.zeros((WARP_SIZE,), dtype=torch.float32, device="cuda")

        expected = (A + B).to(dtype=torch.float32, device="cpu")
        kernel_add_nested[lambda: ((1, 1, 1), (64, 1, 1))](A, B, C)

        actual = C.cpu()

        self.assertTrue(
            torch.allclose(actual, expected),
            f"Expected: {expected.tolist()}, Actual: {actual.tolist()}",
        )

    def test_nested_function_shadows_global(self):
        @avelang.jit
        def kernel_shadow(
            a: S.Tensor((64,), S.f32),
            b: S.Tensor((64,), S.f32),
            c: S.Tensor((64,), S.f32),
        ):
            def helper(x: S.f32, y: S.f32) -> S.f32:
                return x + y

            tid = S.thread_id(0)
            c[tid] = helper(a[tid], b[tid])

        self.assertIsInstance(kernel_shadow.cache_key, str)


if __name__ == "__main__":
    unittest.main()
