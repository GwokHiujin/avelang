#!/usr/bin/env python3
import os
import unittest

import torch

import avelang
import avelang.language as S

WGMMA_SWIZZLE_128B = S.WGMMA_SWIZZLE_128B

TILE_M = 64
TILE_N = 64
TILE_K = 64
K_UNROLL = 8
THREADS_PER_CTA = 128
BF16_BYTES = 2
CP_ASYNC_BYTES = 16
BF16_PER_CP_ASYNC = CP_ASYNC_BYTES // BF16_BYTES
CP_ASYNC_CHUNKS_PER_ROW = TILE_K // BF16_PER_CP_ASYNC
CP_ASYNC_CHUNKS = (TILE_M * TILE_K) // BF16_PER_CP_ASYNC
STORE_ITERS = (TILE_M * TILE_N) // THREADS_PER_CTA

SQUARE_GEMM_SIZES = (1024, 2048, 4096, 8192, 16384)


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
def gemm_wgmma_bf16_square_kernel(
    a_ptr: S.Pointer(S.bf16),
    b_ptr: S.Pointer(S.bf16),
    c_ptr: S.Pointer(S.bf16),
    size: S.constexpr,
):
    tid = S.thread_id(0)
    block_m = S.block_id(1) * TILE_M
    block_n = S.block_id(0) * TILE_N

    a = S.make_tensor(a_ptr, S.bf16, S.make_layout((size, size), (size, 1)))
    b = S.make_tensor(b_ptr, S.bf16, S.make_layout((size, size), (size, 1)))
    c = S.make_tensor(c_ptr, S.bf16, S.make_layout((size, size), (size, 1)))

    a_shared = S.make_shared((TILE_M, TILE_K), S.bf16)
    b_shared = S.make_shared((TILE_K, TILE_N), S.bf16)
    c_shared = S.make_shared((TILE_M, TILE_N), S.f32)
    c_accum_shared = S.make_shared((TILE_M, TILE_N), S.f32)

    k_groups = size // (TILE_K * K_UNROLL)

    for i in S.range(STORE_ITERS):
        idx = tid + i * THREADS_PER_CTA
        row = idx // TILE_N
        col = idx % TILE_N
        c_accum_shared[row, col] = S.convert(0.0, S.f32)

    S.syncthreads()

    for k_group in S.range(k_groups):
        # Keep the WGMMA accumulator in registers across 8 64-wide K stages 
        k_base = k_group * TILE_K * K_UNROLL

        # Each CTA stages a 64x64 BF16 A tile and a 64x64 BF16 B tile.
        # The destination offsets write directly into the WGMMA 128B swizzled
        # shared-memory layout expected by make_wgmma_descriptor.
        acc = S.nvvm.wgmma_init_accumulator(TILE_M, TILE_N)

        for i in S.range(CP_ASYNC_CHUNKS // THREADS_PER_CTA):
            chunk = tid + i * THREADS_PER_CTA
            row = chunk // CP_ASYNC_CHUNKS_PER_ROW
            col = (chunk % CP_ASYNC_CHUNKS_PER_ROW) * BF16_PER_CP_ASYNC
            swizzled_col = col ^ ((row % 8) * 8)
            dst_offset = (row * TILE_K + swizzled_col) * BF16_BYTES

            a_src_offset = ((block_m + row) * size + k_base + col) * BF16_BYTES
            b_src_offset = ((k_base + row) * size + block_n + col) * BF16_BYTES

            S.nvvm.cp_async_ca_shared_global(
                a_shared, a, dst_offset, a_src_offset, CP_ASYNC_BYTES
            )
            S.nvvm.cp_async_ca_shared_global(
                b_shared, b, dst_offset, b_src_offset, CP_ASYNC_BYTES
            )

        S.nvvm.cp_async_commit_group()
        S.nvvm.cp_async_wait_group(0)
        S.syncthreads()

        desc_a = S.nvvm.make_wgmma_descriptor(
            a_shared, WGMMA_SWIZZLE_128B, 0, 0, 0
        )
        desc_b = S.nvvm.make_wgmma_descriptor(
            b_shared, WGMMA_SWIZZLE_128B, 0, 0, 0
        )
        acc = S.nvvm.wgmma_async(desc_a, desc_b, acc)
        S.syncthreads()

        k_base = k_base + TILE_K

        for i in S.range(CP_ASYNC_CHUNKS // THREADS_PER_CTA):
            chunk = tid + i * THREADS_PER_CTA
            row = chunk // CP_ASYNC_CHUNKS_PER_ROW
            col = (chunk % CP_ASYNC_CHUNKS_PER_ROW) * BF16_PER_CP_ASYNC
            swizzled_col = col ^ ((row % 8) * 8)
            dst_offset = (row * TILE_K + swizzled_col) * BF16_BYTES

            a_src_offset = ((block_m + row) * size + k_base + col) * BF16_BYTES
            b_src_offset = ((k_base + row) * size + block_n + col) * BF16_BYTES

            S.nvvm.cp_async_ca_shared_global(
                a_shared, a, dst_offset, a_src_offset, CP_ASYNC_BYTES
            )
            S.nvvm.cp_async_ca_shared_global(
                b_shared, b, dst_offset, b_src_offset, CP_ASYNC_BYTES
            )

        S.nvvm.cp_async_commit_group()
        S.nvvm.cp_async_wait_group(0)
        S.syncthreads()

        desc_a = S.nvvm.make_wgmma_descriptor(
            a_shared, WGMMA_SWIZZLE_128B, 0, 0, 0
        )
        desc_b = S.nvvm.make_wgmma_descriptor(
            b_shared, WGMMA_SWIZZLE_128B, 0, 0, 0
        )
        acc = S.nvvm.wgmma_async(desc_a, desc_b, acc)
        S.syncthreads()

        k_base = k_base + TILE_K

        for i in S.range(CP_ASYNC_CHUNKS // THREADS_PER_CTA):
            chunk = tid + i * THREADS_PER_CTA
            row = chunk // CP_ASYNC_CHUNKS_PER_ROW
            col = (chunk % CP_ASYNC_CHUNKS_PER_ROW) * BF16_PER_CP_ASYNC
            swizzled_col = col ^ ((row % 8) * 8)
            dst_offset = (row * TILE_K + swizzled_col) * BF16_BYTES

            a_src_offset = ((block_m + row) * size + k_base + col) * BF16_BYTES
            b_src_offset = ((k_base + row) * size + block_n + col) * BF16_BYTES

            S.nvvm.cp_async_ca_shared_global(
                a_shared, a, dst_offset, a_src_offset, CP_ASYNC_BYTES
            )
            S.nvvm.cp_async_ca_shared_global(
                b_shared, b, dst_offset, b_src_offset, CP_ASYNC_BYTES
            )

        S.nvvm.cp_async_commit_group()
        S.nvvm.cp_async_wait_group(0)
        S.syncthreads()

        desc_a = S.nvvm.make_wgmma_descriptor(
            a_shared, WGMMA_SWIZZLE_128B, 0, 0, 0
        )
        desc_b = S.nvvm.make_wgmma_descriptor(
            b_shared, WGMMA_SWIZZLE_128B, 0, 0, 0
        )
        acc = S.nvvm.wgmma_async(desc_a, desc_b, acc)
        S.syncthreads()

        k_base = k_base + TILE_K

        for i in S.range(CP_ASYNC_CHUNKS // THREADS_PER_CTA):
            chunk = tid + i * THREADS_PER_CTA
            row = chunk // CP_ASYNC_CHUNKS_PER_ROW
            col = (chunk % CP_ASYNC_CHUNKS_PER_ROW) * BF16_PER_CP_ASYNC
            swizzled_col = col ^ ((row % 8) * 8)
            dst_offset = (row * TILE_K + swizzled_col) * BF16_BYTES

            a_src_offset = ((block_m + row) * size + k_base + col) * BF16_BYTES
            b_src_offset = ((k_base + row) * size + block_n + col) * BF16_BYTES

            S.nvvm.cp_async_ca_shared_global(
                a_shared, a, dst_offset, a_src_offset, CP_ASYNC_BYTES
            )
            S.nvvm.cp_async_ca_shared_global(
                b_shared, b, dst_offset, b_src_offset, CP_ASYNC_BYTES
            )

        S.nvvm.cp_async_commit_group()
        S.nvvm.cp_async_wait_group(0)
        S.syncthreads()

        desc_a = S.nvvm.make_wgmma_descriptor(
            a_shared, WGMMA_SWIZZLE_128B, 0, 0, 0
        )
        desc_b = S.nvvm.make_wgmma_descriptor(
            b_shared, WGMMA_SWIZZLE_128B, 0, 0, 0
        )
        acc = S.nvvm.wgmma_async(desc_a, desc_b, acc)
        S.syncthreads()

        k_base = k_base + TILE_K

        for i in S.range(CP_ASYNC_CHUNKS // THREADS_PER_CTA):
            chunk = tid + i * THREADS_PER_CTA
            row = chunk // CP_ASYNC_CHUNKS_PER_ROW
            col = (chunk % CP_ASYNC_CHUNKS_PER_ROW) * BF16_PER_CP_ASYNC
            swizzled_col = col ^ ((row % 8) * 8)
            dst_offset = (row * TILE_K + swizzled_col) * BF16_BYTES

            a_src_offset = ((block_m + row) * size + k_base + col) * BF16_BYTES
            b_src_offset = ((k_base + row) * size + block_n + col) * BF16_BYTES

            S.nvvm.cp_async_ca_shared_global(
                a_shared, a, dst_offset, a_src_offset, CP_ASYNC_BYTES
            )
            S.nvvm.cp_async_ca_shared_global(
                b_shared, b, dst_offset, b_src_offset, CP_ASYNC_BYTES
            )

        S.nvvm.cp_async_commit_group()
        S.nvvm.cp_async_wait_group(0)
        S.syncthreads()

        desc_a = S.nvvm.make_wgmma_descriptor(
            a_shared, WGMMA_SWIZZLE_128B, 0, 0, 0
        )
        desc_b = S.nvvm.make_wgmma_descriptor(
            b_shared, WGMMA_SWIZZLE_128B, 0, 0, 0
        )
        acc = S.nvvm.wgmma_async(desc_a, desc_b, acc)
        S.syncthreads()

        k_base = k_base + TILE_K

        for i in S.range(CP_ASYNC_CHUNKS // THREADS_PER_CTA):
            chunk = tid + i * THREADS_PER_CTA
            row = chunk // CP_ASYNC_CHUNKS_PER_ROW
            col = (chunk % CP_ASYNC_CHUNKS_PER_ROW) * BF16_PER_CP_ASYNC
            swizzled_col = col ^ ((row % 8) * 8)
            dst_offset = (row * TILE_K + swizzled_col) * BF16_BYTES

            a_src_offset = ((block_m + row) * size + k_base + col) * BF16_BYTES
            b_src_offset = ((k_base + row) * size + block_n + col) * BF16_BYTES

            S.nvvm.cp_async_ca_shared_global(
                a_shared, a, dst_offset, a_src_offset, CP_ASYNC_BYTES
            )
            S.nvvm.cp_async_ca_shared_global(
                b_shared, b, dst_offset, b_src_offset, CP_ASYNC_BYTES
            )

        S.nvvm.cp_async_commit_group()
        S.nvvm.cp_async_wait_group(0)
        S.syncthreads()

        desc_a = S.nvvm.make_wgmma_descriptor(
            a_shared, WGMMA_SWIZZLE_128B, 0, 0, 0
        )
        desc_b = S.nvvm.make_wgmma_descriptor(
            b_shared, WGMMA_SWIZZLE_128B, 0, 0, 0
        )
        acc = S.nvvm.wgmma_async(desc_a, desc_b, acc)
        S.syncthreads()

        k_base = k_base + TILE_K

        for i in S.range(CP_ASYNC_CHUNKS // THREADS_PER_CTA):
            chunk = tid + i * THREADS_PER_CTA
            row = chunk // CP_ASYNC_CHUNKS_PER_ROW
            col = (chunk % CP_ASYNC_CHUNKS_PER_ROW) * BF16_PER_CP_ASYNC
            swizzled_col = col ^ ((row % 8) * 8)
            dst_offset = (row * TILE_K + swizzled_col) * BF16_BYTES

            a_src_offset = ((block_m + row) * size + k_base + col) * BF16_BYTES
            b_src_offset = ((k_base + row) * size + block_n + col) * BF16_BYTES

            S.nvvm.cp_async_ca_shared_global(
                a_shared, a, dst_offset, a_src_offset, CP_ASYNC_BYTES
            )
            S.nvvm.cp_async_ca_shared_global(
                b_shared, b, dst_offset, b_src_offset, CP_ASYNC_BYTES
            )

        S.nvvm.cp_async_commit_group()
        S.nvvm.cp_async_wait_group(0)
        S.syncthreads()

        desc_a = S.nvvm.make_wgmma_descriptor(
            a_shared, WGMMA_SWIZZLE_128B, 0, 0, 0
        )
        desc_b = S.nvvm.make_wgmma_descriptor(
            b_shared, WGMMA_SWIZZLE_128B, 0, 0, 0
        )
        acc = S.nvvm.wgmma_async(desc_a, desc_b, acc)
        S.syncthreads()

        k_base = k_base + TILE_K

        for i in S.range(CP_ASYNC_CHUNKS // THREADS_PER_CTA):
            chunk = tid + i * THREADS_PER_CTA
            row = chunk // CP_ASYNC_CHUNKS_PER_ROW
            col = (chunk % CP_ASYNC_CHUNKS_PER_ROW) * BF16_PER_CP_ASYNC
            swizzled_col = col ^ ((row % 8) * 8)
            dst_offset = (row * TILE_K + swizzled_col) * BF16_BYTES

            a_src_offset = ((block_m + row) * size + k_base + col) * BF16_BYTES
            b_src_offset = ((k_base + row) * size + block_n + col) * BF16_BYTES

            S.nvvm.cp_async_ca_shared_global(
                a_shared, a, dst_offset, a_src_offset, CP_ASYNC_BYTES
            )
            S.nvvm.cp_async_ca_shared_global(
                b_shared, b, dst_offset, b_src_offset, CP_ASYNC_BYTES
            )

        S.nvvm.cp_async_commit_group()
        S.nvvm.cp_async_wait_group(0)
        S.syncthreads()

        desc_a = S.nvvm.make_wgmma_descriptor(
            a_shared, WGMMA_SWIZZLE_128B, 0, 0, 0
        )
        desc_b = S.nvvm.make_wgmma_descriptor(
            b_shared, WGMMA_SWIZZLE_128B, 0, 0, 0
        )
        acc = S.nvvm.wgmma_async(desc_a, desc_b, acc)
        S.nvvm.wgmma_store(acc, c_shared)
        S.syncthreads()

        for i in S.range(STORE_ITERS):
            idx = tid + i * THREADS_PER_CTA
            row = idx // TILE_N
            col = idx % TILE_N
            c_accum_shared[row, col] = c_accum_shared[row, col] + c_shared[row, col]

        S.syncthreads()

    for i in S.range(STORE_ITERS):
        idx = tid + i * THREADS_PER_CTA
        row = idx // TILE_N
        col = idx % TILE_N
        c[block_m + row, block_n + col] = S.convert(c_accum_shared[row, col], S.bf16)


@unittest.skipUnless(
    get_wgmma_device() is not None,
    "Requires CUDA on an NVIDIA Hopper-or-newer GPU with WGMMA support.",
)
class TestWgmmaBf16SquareGemm(unittest.TestCase):
    def setUp(self):
        device_idx = get_wgmma_device()
        self.assertIsNotNone(device_idx)
        torch.cuda.set_device(device_idx)
        self.device = torch.device(f"cuda:{device_idx}")

    def _run_square_ones_case(self, size: int):
        self.assertEqual(size % TILE_M, 0)
        a = torch.ones((size, size), dtype=torch.bfloat16, device=self.device)
        b = torch.ones((size, size), dtype=torch.bfloat16, device=self.device)
        c = torch.zeros((size, size), dtype=torch.bfloat16, device=self.device)

        grid = (size // TILE_N, size // TILE_M, 1)
        block = (THREADS_PER_CTA, 1, 1)
        gemm_wgmma_bf16_square_kernel[lambda: (grid, block)](a, b, c, size)
        torch.cuda.synchronize(self.device)

        max_diff = c.float().sub_(float(size)).abs_().max().item()
        self.assertEqual(
            max_diff,
            0.0,
            msg=(
                f"WGMMA BF16 square GEMM mismatch for size {size}. "
                f"Max absolute difference: {max_diff}"
            ),
        )

    def test_wgmma_bf16_square_gemm_1024_random_reference(self):
        size = 1024
        torch.manual_seed(0)
        a = torch.randn((size, size), dtype=torch.bfloat16, device=self.device)
        b = torch.randn((size, size), dtype=torch.bfloat16, device=self.device)
        c = torch.zeros((size, size), dtype=torch.bfloat16, device=self.device)

        grid = (size // TILE_N, size // TILE_M, 1)
        block = (THREADS_PER_CTA, 1, 1)
        gemm_wgmma_bf16_square_kernel[lambda: (grid, block)](a, b, c, size)
        torch.cuda.synchronize(self.device)

        expected = (a.float() @ b.float()).to(dtype=torch.bfloat16)
        actual = c
        max_diff = torch.max(torch.abs(actual.float() - expected.float()))
        self.assertTrue(
            torch.allclose(actual, expected, rtol=1e-1, atol=1e-1),
            msg=(
                "WGMMA BF16 random GEMM mismatch for size 1024. "
                f"Max absolute difference: {max_diff}"
            ),
        )

    def test_wgmma_bf16_square_gemm_cublaslt_artifact_sizes(self):
        for size in SQUARE_GEMM_SIZES:
            with self.subTest(size=size):
                self._run_square_ones_case(size)


if __name__ == "__main__":
    unittest.main()
