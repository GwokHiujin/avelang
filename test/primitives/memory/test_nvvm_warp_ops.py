#!/usr/bin/env python3
import unittest

import torch

import avelang
import avelang.language as S


def get_hopper_device():
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return None
    for device_idx in range(torch.cuda.device_count()):
        try:
            major, _minor = torch.cuda.get_device_capability(device_idx)
            if major < 9:
                continue
            torch.cuda.set_device(device_idx)
            torch.empty(1, device=f"cuda:{device_idx}")
            torch.cuda.synchronize(device_idx)
            return device_idx
        except Exception:
            continue
    return None


@avelang.jit
def kernel_setmaxnreg(out: S.Tensor((128,), S.i32)):
    tid = S.thread_id(0)
    S.nvvm.setmaxnreg_dec(24)
    S.nvvm.setmaxnreg_inc(32)
    out[tid] = S.convert(tid, S.i32)


@avelang.jit
def kernel_elect_sync(out: S.Tensor((128,), S.i32)):
    tid = S.thread_id(0)
    out[tid] = S.convert(S.nvvm.elect_sync(), S.i32)


@avelang.jit
def kernel_named_barrier(
    out: S.Tensor((64,), S.i32), barrier_id: S.i32, thread_count: S.i32
):
    tid = S.thread_id(0)
    shared = S.make_shared((64,), S.i32)
    shared[tid] = S.convert(tid, S.i32)
    if tid < 32:
        S.nvvm.named_barrier_arrive(barrier_id, thread_count)
        out[tid] = shared[tid]
    else:
        S.nvvm.named_barrier_sync(barrier_id, thread_count)
        out[tid] = shared[tid - 32]


@avelang.jit
def kernel_syncwarp(out: S.Tensor((32,), S.i32), mask: S.i32):
    tid = S.thread_id(0)
    shared = S.make_shared((32,), S.i32)
    shared[tid] = S.convert(tid, S.i32)
    S.nvvm.syncwarp(mask)
    out[tid] = shared[(tid + 1) % 32]


@unittest.skipUnless(
    get_hopper_device() is not None,
    "Requires CUDA on an NVIDIA Hopper-or-newer GPU.",
)
class TestNVVMWarpOps(unittest.TestCase):
    def test_setmaxnreg(self):
        device_idx = get_hopper_device()
        assert device_idx is not None
        torch.cuda.set_device(device_idx)
        device = torch.device(f"cuda:{device_idx}")
        out = torch.zeros((128,), dtype=torch.int32, device=device)

        kernel_setmaxnreg[lambda: ((1, 1, 1), (128, 1, 1))](out)

        expected = torch.arange(128, dtype=torch.int32)
        self.assertTrue(torch.equal(out.cpu(), expected))

    def test_elect_sync(self):
        device_idx = get_hopper_device()
        assert device_idx is not None
        torch.cuda.set_device(device_idx)
        device = torch.device(f"cuda:{device_idx}")
        out = torch.zeros((128,), dtype=torch.int32, device=device)

        kernel_elect_sync[lambda: ((1, 1, 1), (128, 1, 1))](out)

        leaders_per_warp = (out.cpu().reshape(4, 32) != 0).sum(dim=1)
        self.assertTrue(torch.equal(leaders_per_warp, torch.ones(4)))

    def test_named_barrier_arrive_and_sync(self):
        device_idx = get_hopper_device()
        assert device_idx is not None
        torch.cuda.set_device(device_idx)
        device = torch.device(f"cuda:{device_idx}")
        out = torch.zeros((64,), dtype=torch.int32, device=device)

        kernel_named_barrier[lambda: ((1, 1, 1), (64, 1, 1))](out, 1, 64)

        expected = torch.arange(32, dtype=torch.int32).repeat(2)
        self.assertTrue(torch.equal(out.cpu(), expected))

    def test_syncwarp(self):
        device_idx = get_hopper_device()
        assert device_idx is not None
        torch.cuda.set_device(device_idx)
        device = torch.device(f"cuda:{device_idx}")
        out = torch.zeros((32,), dtype=torch.int32, device=device)

        kernel_syncwarp[lambda: ((1, 1, 1), (32, 1, 1))](out, -1)

        expected = torch.roll(torch.arange(32, dtype=torch.int32), -1)
        self.assertTrue(torch.equal(out.cpu(), expected))


if __name__ == "__main__":
    unittest.main()
