#!/usr/bin/env python3
import unittest

import torch

import avelang
import avelang.language as S


@avelang.jit
def tuple_add(x: S.u32, y: S.u32) -> (S.u32, S.u32):
    return x + 1, y + 2


@avelang.jit
def kernel_tuple_return(
    a: S.Tensor((1,), S.u32),
    b: S.Tensor((1,), S.u32),
    out: S.Tensor((2,), S.u32),
):
    x = a[0]
    y = b[0]
    id_m, id_n = tuple_add(x, y)
    out[0] = id_m
    out[1] = id_n


class TestTupleReturns(unittest.TestCase):
    def test_tuple_return_assignment(self):
        a = torch.tensor([5], dtype=torch.int32, device="cuda")
        b = torch.tensor([7], dtype=torch.int32, device="cuda")
        out = torch.zeros((2,), dtype=torch.int32, device="cuda")

        kernel_tuple_return[lambda: ((1, 1, 1), (1, 1, 1))](a, b, out)

        expected = torch.tensor([6, 9], dtype=torch.int32, device="cuda")
        self.assertTrue(
            torch.equal(out.cpu(), expected.cpu()),
            f"Expected: {expected.tolist()}, Actual: {out.tolist()}",
        )


if __name__ == "__main__":
    unittest.main()
