#!/usr/bin/env python3
import unittest

import torch

import avelang
import avelang.language as S


@avelang.jit
def kernel_dyn_memset(src: S.Pointer(S.i32), v: S.i32, n: S.u32):
    s = S.make_tensor(src, S.i32, S.make_layout((n,), (1,)))
    tid = S.thread_id(0)
    for i in S.range((n + S.block_dim(0) - 1) // S.block_dim(0)):
        idx = tid + i * S.block_dim(0)
        if idx < n:
            s[idx] = v


class TestMakeTensorFromPointer(unittest.TestCase):
    def test_dyn_memset(self):
        N = 256
        out = torch.zeros((N,), dtype=torch.int32, device="cuda")

        kernel_dyn_memset[lambda: ((1, 1, 1), (64, 1, 1))](out, 42, N)

        self.assertTrue(torch.equal(out, torch.full((N,), 42, dtype=torch.int32, device="cuda")))


if __name__ == "__main__":
    unittest.main()
