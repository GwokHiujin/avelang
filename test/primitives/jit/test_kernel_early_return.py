#!/usr/bin/env python3
import unittest

import torch

import avelang
import avelang.language as S


@avelang.jit
def kernel_guard_return(out: S.Tensor((1, 2, 16), S.i32)):
    row = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    if row >= 16:
        return

    acc = S.convert(0, S.i32)
    for key_idx in S.range(16):
        if key_idx <= row:
            acc = acc + 1

    out[S.block_id(2), S.block_id(1), row] = acc


class TestKernelEarlyReturn(unittest.TestCase):
    def test_guard_return_does_not_clobber_adjacent_heads(self):
        out = torch.full((1, 2, 16), -1, dtype=torch.int32, device="cuda")
        expected_row = torch.arange(1, 17, dtype=torch.int32)
        expected = torch.stack((expected_row, expected_row), dim=0).unsqueeze(0)

        kernel_guard_return[lambda: ((1, 2, 1), (64, 1, 1))](out)

        self.assertTrue(
            torch.equal(out.cpu(), expected),
            msg=f"Expected:\n{expected}\nActual:\n{out.cpu()}",
        )


if __name__ == "__main__":
    unittest.main()
