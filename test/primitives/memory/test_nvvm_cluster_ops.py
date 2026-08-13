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
def kernel_cluster_sync(out: S.Tensor((32,), S.i32)):
    tid = S.thread_id(0)
    S.nvvm.fence_mbarrier_init_release_cluster()
    S.nvvm.cluster_arrive_relaxed()
    S.nvvm.cluster_wait()
    out[tid] = S.convert(tid, S.i32)


@avelang.jit
def kernel_cluster_rank(out: S.Tensor((32,), S.i32)):
    tid = S.thread_id(0)
    out[tid] = S.nvvm.cluster_block_rank()


@avelang.jit
def kernel_cluster_rank_pair(out: S.Tensor((2,), S.i32)):
    if S.thread_id(0) == 0:
        out[S.block_id(0)] = S.nvvm.cluster_block_rank()


@avelang.jit
def kernel_griddepcontrol_wait(out: S.Tensor((32,), S.i32)):
    tid = S.thread_id(0)
    S.nvvm.griddepcontrol_wait()
    out[tid] = S.convert(tid, S.i32)


@avelang.jit
def kernel_mbarrier_arrive_cluster(out: S.Tensor((32,), S.i32)):
    tid = S.thread_id(0)
    barrier = S.nvvm.mbarrier_create()
    S.nvvm.mbarrier_init(barrier, 0, 1, tid == 0)
    if tid == 0:
        S.nvvm.mbarrier_arrive_cluster(barrier, 0, 0)
    S.nvvm.mbarrier_try_wait_parity(barrier, 0, 10000000, 0)
    out[tid] = S.convert(tid, S.i32)


@unittest.skipUnless(
    get_hopper_device() is not None,
    "Requires CUDA on an NVIDIA Hopper-or-newer GPU.",
)
class TestNVVMClusterOps(unittest.TestCase):
    def test_cluster_sync_single_cta_cluster(self):
        device_idx = get_hopper_device()
        assert device_idx is not None
        torch.cuda.set_device(device_idx)
        device = torch.device(f"cuda:{device_idx}")
        out = torch.zeros((32,), dtype=torch.int32, device=device)

        kernel_cluster_sync[lambda: ((1, 1, 1), (32, 1, 1))](out)

        self.assertTrue(torch.equal(out.cpu(), torch.arange(32)))

    def test_cluster_rank_single_cta_cluster(self):
        device_idx = get_hopper_device()
        assert device_idx is not None
        torch.cuda.set_device(device_idx)
        device = torch.device(f"cuda:{device_idx}")
        out = torch.full((32,), -1, dtype=torch.int32, device=device)

        kernel_cluster_rank[lambda: ((1, 1, 1), (32, 1, 1))](out)

        self.assertTrue(torch.equal(out.cpu(), torch.zeros(32, dtype=torch.int32)))

    def test_cluster_rank_two_cta_cluster(self):
        device_idx = get_hopper_device()
        assert device_idx is not None
        torch.cuda.set_device(device_idx)
        device = torch.device(f"cuda:{device_idx}")
        out = torch.full((2,), -1, dtype=torch.int32, device=device)

        kernel_cluster_rank_pair[
            lambda: ((2, 1, 1), (32, 1, 1), (2, 1, 1))
        ](out)

        self.assertTrue(torch.equal(out.cpu(), torch.arange(2)))

    def test_griddepcontrol_wait(self):
        device_idx = get_hopper_device()
        assert device_idx is not None
        torch.cuda.set_device(device_idx)
        device = torch.device(f"cuda:{device_idx}")
        out = torch.full((32,), -1, dtype=torch.int32, device=device)

        kernel_griddepcontrol_wait[
            lambda: ((1, 1, 1), (32, 1, 1))
        ](out)

        self.assertTrue(torch.equal(out.cpu(), torch.arange(32)))

    def test_mbarrier_arrive_cluster_local_rank(self):
        device_idx = get_hopper_device()
        assert device_idx is not None
        torch.cuda.set_device(device_idx)
        device = torch.device(f"cuda:{device_idx}")
        out = torch.full((32,), -1, dtype=torch.int32, device=device)

        kernel_mbarrier_arrive_cluster[
            lambda: ((1, 1, 1), (32, 1, 1))
        ](out)

        self.assertTrue(torch.equal(out.cpu(), torch.arange(32)))


if __name__ == "__main__":
    unittest.main()
