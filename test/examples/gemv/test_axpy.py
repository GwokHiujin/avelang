#!/usr/bin/env python3
import unittest

import torch

import avelang
import avelang.language as al


@avelang.jit
def axpy(
    a: al.i32,
    b: al.i32,
    x: al.Tensor((32,), al.i32),
    y: al.Tensor((32,), al.i32),
):
    idx = al.block_id(0) * al.block_dim(0) + al.thread_id(0)
    y[idx] = a * x[idx] + b


class TestAXPY(unittest.TestCase):
    def setUp(self):
        """alet up test fixtures before each test method."""
        torch.manual_seed(0)
        self.N = 32

    def test_axpy(self):
        """[NVVM] Test integer AXPY kernel."""
        x = torch.randint(0, 100, (self.N,)).to(dtype=torch.int32, device="cuda")
        y = torch.zeros(self.N, dtype=torch.int32, device="cuda")
        a = 1
        b = 0
        expected = (a * x + b).to(dtype=torch.int32, device="cpu")

        axpy[lambda: ((1, 1, 1), (self.N, 1, 1))](a, b, x, y)

        actual = y.to("cpu")

        self.assertTrue(
            torch.equal(actual, expected),
            msg=f"AXPY results do not match.\nExpected:\n{expected}\nActual:\n{actual}\n"
            f"Max absolute difference: {torch.max(torch.abs(actual - expected))}",
        )


if __name__ == "__main__":
    unittest.main()
