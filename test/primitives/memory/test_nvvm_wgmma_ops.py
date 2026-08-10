#!/usr/bin/env python3
import unittest

import torch

import avelang
import avelang.language as S

WGMMA_SWIZZLE_32B = S.WGMMA_SWIZZLE_32B
WGMMA_SWIZZLE_128B = S.WGMMA_SWIZZLE_128B


def get_wgmma_device():
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
def kernel_nvvm_wgmma_sync_primitives(out: S.Tensor((128,), S.i32)):
    tid = S.thread_id(0)

    S.nvvm.wgmma_fence_aligned()
    S.nvvm.wgmma_group_sync_aligned()
    S.nvvm.wgmma_wait_group_sync(0)

    out[tid] = S.convert(tid, S.i32)


@avelang.jit
def kernel_nvvm_wgmma(
    a: S.Tensor((64, 16), S.f16),
    b: S.Tensor((16, 16), S.f16),
    out: S.Tensor((64, 16), S.f32),
):
    tid = S.thread_id(0)

    a_stage = S.make_shared((64, 16), S.f16)
    b_stage = S.make_shared((16, 16), S.f16)
    a_shared = S.make_shared((64, 16), S.f16, 128)
    b_shared = S.make_shared((16, 16), S.f16, 128)
    c_shared = S.make_shared((64, 16), S.f32)

    a_offset_bytes = tid * 16
    S.nvvm.cp_async_ca_shared_global(a_stage, a, a_offset_bytes, a_offset_bytes, 16)

    if tid < 32:
        b_offset_bytes = tid * 16
        S.nvvm.cp_async_ca_shared_global(
            b_stage, b, b_offset_bytes, b_offset_bytes, 16
        )

    S.nvvm.cp_async_commit_group()
    S.nvvm.cp_async_wait_group(0)
    S.syncthreads()

    for i in S.range(8):
        linear_idx_a = tid + i * 128
        row_a = linear_idx_a // 16
        col_a = linear_idx_a % 16
        swizzle_a = (row_a // 4) % 2
        swizzled_col_a = col_a
        if swizzle_a == 1:
            if col_a < 8:
                swizzled_col_a = col_a + 8
            else:
                swizzled_col_a = col_a - 8
        a_shared[row_a, swizzled_col_a] = a_stage[row_a, col_a]

    for i in S.range(2):
        linear_idx_b = tid + i * 128
        row_b = linear_idx_b // 16
        col_b = linear_idx_b % 16
        swizzle_b = (row_b // 4) % 2
        swizzled_col_b = col_b
        if swizzle_b == 1:
            if col_b < 8:
                swizzled_col_b = col_b + 8
            else:
                swizzled_col_b = col_b - 8
        b_shared[row_b, swizzled_col_b] = b_stage[row_b, col_b]

    S.syncthreads()

    desc_a = S.nvvm.make_wgmma_descriptor(a_shared, WGMMA_SWIZZLE_32B, 0, 0, 0)
    desc_b = S.nvvm.make_wgmma_descriptor(b_shared, WGMMA_SWIZZLE_32B, 0, 0, 0)
    acc = S.nvvm.wgmma_init_accumulator(64, 16)
    result = S.nvvm.wgmma_async(desc_a, desc_b, acc)
    S.nvvm.wgmma_store(result, c_shared)

    S.syncthreads()

    for i in S.range(8):
        linear_idx_c = tid + i * 128
        row_c = linear_idx_c // 16
        col_c = linear_idx_c % 16
        out[row_c, col_c] = c_shared[row_c, col_c]


@avelang.jit
def kernel_nvvm_wgmma_swizzle_128b(
    a: S.Tensor((64, 64), S.f16),
    b: S.Tensor((64, 64), S.f16),
    out: S.Tensor((64, 64), S.f32),
):
    tid = S.thread_id(0)

    a_stage = S.make_shared((64, 64), S.f16)
    b_stage = S.make_shared((64, 64), S.f16)
    a_shared = S.make_shared((64, 64), S.f16, 128)
    b_shared = S.make_shared((64, 64), S.f16, 128)
    c_shared = S.make_shared((64, 64), S.f32)

    # 128 threads x 16B x 4 = 8192B covers 64x64xf16.
    for i in S.range(4):
        off_bytes = (tid + i * 128) * 16
        S.nvvm.cp_async_ca_shared_global(a_stage, a, off_bytes, off_bytes, 16)
        S.nvvm.cp_async_ca_shared_global(b_stage, b, off_bytes, off_bytes, 16)

    S.nvvm.cp_async_commit_group()
    S.nvvm.cp_async_wait_group(0)
    S.syncthreads()

    # Swizzle-128B layout transform for f16 shared inputs:
    # col' = col XOR ((row & 0x7) * 8)
    for i in S.range(32):
        idx = tid + i * 128
        row = idx // 64
        col = idx % 64
        swizzle = (row % 8) * 8
        swizzled_col = col ^ swizzle
        a_shared[row, swizzled_col] = a_stage[row, col]
        b_shared[row, swizzled_col] = b_stage[row, col]

    S.syncthreads()

    desc_a = S.nvvm.make_wgmma_descriptor(a_shared, WGMMA_SWIZZLE_128B, 0, 0, 0)
    desc_b = S.nvvm.make_wgmma_descriptor(b_shared, WGMMA_SWIZZLE_128B, 0, 0, 0)
    acc = S.nvvm.wgmma_init_accumulator(64, 64)
    result = S.nvvm.wgmma_async(desc_a, desc_b, acc)
    S.nvvm.wgmma_store(result, c_shared)

    S.syncthreads()

    for i in S.range(32):
        idx = tid + i * 128
        row = idx // 64
        col = idx % 64
        out[row, col] = c_shared[row, col]


@avelang.jit
def kernel_nvvm_raw_wgmma_bf16(
    a: S.Tensor((64, 16), S.bf16),
    b: S.Tensor((16, 64), S.bf16),
    out: S.Tensor((128, 32), S.f32),
):
    tid = S.thread_id(0)
    a_shared = S.make_shared((64, 16), S.bf16, 128)
    b_shared = S.make_shared((16, 64), S.bf16, 128)

    for i in S.range(8):
        idx = tid + i * 128
        row_a = idx // 16
        col_a = idx % 16
        swizzled_col_a = col_a
        if (row_a // 4) % 2 == 1:
            if col_a < 8:
                swizzled_col_a = col_a + 8
            else:
                swizzled_col_a = col_a - 8
        a_shared[row_a, swizzled_col_a] = a[row_a, col_a]

        row_b = idx // 64
        col_b = idx % 64
        swizzled_col_b = col_b
        if (row_b // 4) % 2 == 1:
            if col_b % 16 < 8:
                swizzled_col_b = col_b + 8
            else:
                swizzled_col_b = col_b - 8
        b_shared[row_b, swizzled_col_b] = b[row_b, col_b]

    S.syncthreads()
    desc_a = S.nvvm.make_wgmma_descriptor_bits(
        a_shared, WGMMA_SWIZZLE_32B, 0, 0, 0
    )
    desc_b = S.nvvm.make_wgmma_descriptor_bits(
        b_shared, WGMMA_SWIZZLE_32B, 0, 0, 0
    )
    result = S.nvvm.wgmma_init_result(32)
    S.nvvm.wgmma_fence_aligned()
    result = S.nvvm.wgmma_m64n64k16_f32_bf16_bf16(
        desc_a, desc_b, result, 0
    )
    S.nvvm.wgmma_group_sync_aligned()
    S.nvvm.wgmma_wait_group_sync(0)

    for i in S.range(32):
        out[tid, i] = result[i]


@avelang.jit
def kernel_nvvm_wgmma_bf16_rs(
    a_regs: S.Tensor((128, 4), S.i32),
    b: S.Tensor((16, 128), S.bf16),
    out: S.Tensor((128, 64), S.f32),
):
    tid = S.thread_id(0)
    b_shared = S.make_shared((16, 128), S.bf16, 128)
    for i in S.range(16):
        idx = tid + i * 128
        row = idx // 128
        col = idx % 128
        swizzled_col = col
        if (row // 4) % 2 == 1:
            if col % 16 < 8:
                swizzled_col = col + 8
            else:
                swizzled_col = col - 8
        b_shared[row, swizzled_col] = b[row, col]

    S.syncthreads()
    desc_b = S.nvvm.make_wgmma_descriptor_bits(
        b_shared, WGMMA_SWIZZLE_32B, 0, 0, 0
    )
    result = S.nvvm.wgmma_init_result(64)
    S.nvvm.wgmma_fence_aligned()
    result = S.nvvm.wgmma_m64n128k16_f32_bf16_bf16_rs(
        a_regs[tid], desc_b, result, 0
    )
    S.nvvm.wgmma_group_sync_aligned()
    S.nvvm.wgmma_wait_group_sync(0)
    for i in S.range(64):
        out[tid, i] = result[i]


@avelang.jit
def kernel_nvvm_wgmma_fp8(
    a: S.Tensor((64, 32), S.u8),
    b: S.Tensor((32, 192), S.u8),
    out: S.Tensor((128, 96), S.f32),
):
    tid = S.thread_id(0)
    a_shared = S.make_shared((64, 32), S.u8, 128)
    b_shared = S.make_shared((32, 192), S.u8, 128)
    for i in S.range(16):
        idx = tid + i * 128
        row = idx // 32
        col = idx % 32
        a_shared[row, col] = a[row, col]
    for i in S.range(48):
        idx = tid + i * 128
        row = idx // 192
        col = idx % 192
        b_shared[row, col] = b[row, col]

    S.syncthreads()
    desc_a = S.nvvm.make_wgmma_descriptor_bits(
        a_shared, WGMMA_SWIZZLE_32B, 0, 0, 0
    )
    desc_b = S.nvvm.make_wgmma_descriptor_bits(
        b_shared, WGMMA_SWIZZLE_32B, 0, 0, 0
    )
    result = S.nvvm.wgmma_init_result(96)
    S.nvvm.wgmma_fence_aligned()
    result = S.nvvm.wgmma_m64n192k32_f32_e4m3_e4m3(
        desc_a, desc_b, result, 0
    )
    result = S.nvvm.wgmma_m64n192k32_f32_e4m3_e4m3(
        desc_a, desc_b, result, 1
    )
    S.nvvm.wgmma_group_sync_aligned()
    S.nvvm.wgmma_wait_group_sync(0)
    for i in S.range(96):
        out[tid, i] = result[i]


@unittest.skipUnless(
    get_wgmma_device() is not None,
    "Requires CUDA on an NVIDIA Hopper-or-newer GPU with WGMMA support.",
)
class TestNVVMWGMMAOps(unittest.TestCase):
    def test_wgmma_sync_primitives(self):
        device_idx = get_wgmma_device()
        self.assertIsNotNone(device_idx)
        torch.cuda.set_device(device_idx)
        device = torch.device(f"cuda:{device_idx}")

        out = torch.zeros((128,), dtype=torch.int32, device=device)
        expected = torch.arange(128, dtype=torch.int32, device=device)

        kernel_nvvm_wgmma_sync_primitives[lambda: ((1, 1, 1), (128, 1, 1))](out)

        actual = out.cpu()
        expected = expected.cpu()

        self.assertTrue(
            torch.equal(actual, expected),
            f"Expected: {expected.tolist()}, Actual: {actual.tolist()}",
        )

    def test_gemm_f32_f16_f16_64x16x16(self):
        device_idx = get_wgmma_device()
        assert device_idx is not None
        torch.cuda.set_device(device_idx)
        device = torch.device(f"cuda:{device_idx}")

        a = torch.randn((64, 16), dtype=torch.float16, device=device)
        b = torch.randn((16, 16), dtype=torch.float16, device=device)
        out = torch.zeros((64, 16), dtype=torch.float32, device=device)

        kernel_nvvm_wgmma[lambda: ((1, 1, 1), (128, 1, 1))](a, b, out)

        actual = out.cpu()
        expected = (a @ b).float().cpu()
        self.assertTrue(
            torch.allclose(actual, expected, rtol=1e-2, atol=1e-2),
            msg=f"GEMM results do not match.\nExpected:\n{expected}\nActual:\n{actual}\n"
            f"Max absolute difference: {torch.max(torch.abs(actual - expected))}",
        )

    def test_gemm_f32_f16_f16_64x64x64_swizzle_128b(self):
        device_idx = get_wgmma_device()
        assert device_idx is not None
        torch.cuda.set_device(device_idx)
        device = torch.device(f"cuda:{device_idx}")

        a = torch.randn((64, 64), dtype=torch.float16, device=device)
        b = torch.randn((64, 64), dtype=torch.float16, device=device)
        out = torch.zeros((64, 64), dtype=torch.float32, device=device)

        kernel_nvvm_wgmma_swizzle_128b[lambda: ((1, 1, 1), (128, 1, 1))](
            a, b, out
        )

        actual = out.cpu()
        expected = (a @ b).float().cpu()
        self.assertTrue(
            torch.allclose(actual, expected, rtol=1e-2, atol=1e-2),
            msg=(
                "WGMMA swizzle_128b GEMM results do not match.\n"
                f"Max absolute difference: {torch.max(torch.abs(actual - expected))}\n"
                f"Expected:\n{expected}\nActual:\n{actual}"
            ),
        )

    def test_raw_wgmma_bf16_result_access(self):
        device_idx = get_wgmma_device()
        assert device_idx is not None
        torch.cuda.set_device(device_idx)
        device = torch.device(f"cuda:{device_idx}")
        a = torch.ones((64, 16), dtype=torch.bfloat16, device=device)
        b = torch.ones((16, 64), dtype=torch.bfloat16, device=device)
        out = torch.zeros((128, 32), dtype=torch.float32, device=device)

        kernel_nvvm_raw_wgmma_bf16[
            lambda: ((1, 1, 1), (128, 1, 1))
        ](a, b, out)

        self.assertTrue(torch.equal(out.cpu(), torch.full((128, 32), 16.0)))

    def test_wgmma_bf16_register_shared(self):
        device_idx = get_wgmma_device()
        assert device_idx is not None
        torch.cuda.set_device(device_idx)
        device = torch.device(f"cuda:{device_idx}")
        packed_bf16_ones = 0x3F803F80
        a_regs = torch.full(
            (128, 4), packed_bf16_ones, dtype=torch.int32, device=device
        )
        b = torch.ones((16, 128), dtype=torch.bfloat16, device=device)
        out = torch.zeros((128, 64), dtype=torch.float32, device=device)

        kernel_nvvm_wgmma_bf16_rs[
            lambda: ((1, 1, 1), (128, 1, 1))
        ](a_regs, b, out)

        self.assertTrue(torch.equal(out.cpu(), torch.full((128, 64), 16.0)))

    def test_wgmma_fp8_e4m3_scale_d(self):
        device_idx = get_wgmma_device()
        assert device_idx is not None
        torch.cuda.set_device(device_idx)
        device = torch.device(f"cuda:{device_idx}")
        # E4M3 encoding of 1.0 (sign=0, exponent=7, mantissa=0).
        a = torch.full((64, 32), 0x38, dtype=torch.uint8, device=device)
        b = torch.full((32, 192), 0x38, dtype=torch.uint8, device=device)
        out = torch.zeros((128, 96), dtype=torch.float32, device=device)

        kernel_nvvm_wgmma_fp8[
            lambda: ((1, 1, 1), (128, 1, 1))
        ](a, b, out)

        # One scale_D=0 operation followed by scale_D=1 accumulates 32 twice.
        self.assertTrue(torch.equal(out.cpu(), torch.full((128, 96), 64.0)))


if __name__ == "__main__":
    unittest.main()
