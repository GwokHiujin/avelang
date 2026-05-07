#!/usr/bin/env python3
import unittest

import torch
import avelang
import avelang.language as S


@avelang.jit
def kernel_shuffle_xor_i32(
    inp: S.Tensor((64,), S.i32),
    out: S.Tensor((64,), S.i32),
):
    tid = S.thread_id(0)
    out[tid] = S.shuffle_xor(inp[tid], 1, 32)


@avelang.jit
def kernel_shuffle_up_i32(
    inp: S.Tensor((64,), S.i32),
    out: S.Tensor((64,), S.i32),
):
    tid = S.thread_id(0)
    out[tid] = S.shuffle_up(inp[tid], 1, 32)


@avelang.jit
def kernel_shuffle_down_i32(
    inp: S.Tensor((64,), S.i32),
    out: S.Tensor((64,), S.i32),
):
    tid = S.thread_id(0)
    out[tid] = S.shuffle_down(inp[tid], 1, 32)


@avelang.jit
def kernel_shuffle_idx_i32(
    inp: S.Tensor((64,), S.i32),
    out: S.Tensor((64,), S.i32),
):
    tid = S.thread_id(0)
    out[tid] = S.shuffle(inp[tid], 7, 32)


class TestShuffle(unittest.TestCase):
    def _warp_lanes(self):
        return torch.arange(64, dtype=torch.int64) % 32

    def _warp_bases(self):
        return torch.arange(64, dtype=torch.int64) - self._warp_lanes()

    def test_shuffle_idx_i32(self):
        inp = torch.arange(64, dtype=torch.int32, device="cuda") * 17 + 3
        out = torch.empty_like(inp)

        kernel_shuffle_idx_i32[lambda: ((1, 1, 1), (64, 1, 1))](inp, out)

        expected = inp.cpu()[self._warp_bases() + 7]
        self.assertTrue(
            torch.equal(out.cpu(), expected),
            f"Expected: {expected.tolist()}, Actual: {out.cpu().tolist()}",
        )

    def test_shuffle_up_i32(self):
        inp = torch.arange(64, dtype=torch.int32, device="cuda") * 17 + 3
        out = torch.empty_like(inp)

        kernel_shuffle_up_i32[lambda: ((1, 1, 1), (64, 1, 1))](inp, out)

        lanes = self._warp_lanes()
        source_lanes = torch.where(lanes == 0, lanes, lanes - 1)
        expected = inp.cpu()[self._warp_bases() + source_lanes]
        self.assertTrue(
            torch.equal(out.cpu(), expected),
            f"Expected: {expected.tolist()}, Actual: {out.cpu().tolist()}",
        )

    def test_shuffle_down_i32(self):
        inp = torch.arange(64, dtype=torch.int32, device="cuda") * 17 + 3
        out = torch.empty_like(inp)

        kernel_shuffle_down_i32[lambda: ((1, 1, 1), (64, 1, 1))](inp, out)

        lanes = self._warp_lanes()
        source_lanes = torch.where(lanes == 31, lanes, lanes + 1)
        expected = inp.cpu()[self._warp_bases() + source_lanes]
        self.assertTrue(
            torch.equal(out.cpu(), expected),
            f"Expected: {expected.tolist()}, Actual: {out.cpu().tolist()}",
        )

    def test_shuffle_xor_i32(self):
        inp = torch.arange(64, dtype=torch.int32, device="cuda") * 17 + 3
        out = torch.empty_like(inp)

        kernel_shuffle_xor_i32[lambda: ((1, 1, 1), (64, 1, 1))](inp, out)

        expected = inp.cpu()[self._warp_bases() + (self._warp_lanes() ^ 1)]
        self.assertTrue(
            torch.equal(out.cpu(), expected),
            f"Expected: {expected.tolist()}, Actual: {out.cpu().tolist()}",
        )


if __name__ == "__main__":
    unittest.main()
