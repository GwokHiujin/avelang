#!/usr/bin/env python3
import unittest

import torch

import avelang
import avelang.language as S
from avelang.testing import has_cuda_nvidia


@avelang.jit
def gemm_nvvm_16x8x16(
    A: S.Tensor((128, 128), S.f16),
    B: S.Tensor((128, 128), S.f16),
    C: S.Tensor((128, 128), S.f16),
):
    MMA_M = 16
    MMA_N = 8
    MMA_K = 16

    WARP_SIZE = 32

    K_tiles = 8  # 128 // 16

    warp_row = S.block_id(1) * MMA_M
    warp_col = S.block_id(0) * MMA_N

    lane_id = S.thread_id(0) % WARP_SIZE

    a_smem = S.make_shared((16, 16), S.f16)
    b_smem = S.make_shared((8, 16), S.f16)
    c_smem = S.make_shared((16, 8), S.f16)

    # Accumulator: each thread has 2 uint32 (which hold 4 f16 values total)
    RC = S.make_local((4,), S.f16)
    for i in S.range(4):
        RC[i] = S.convert(0, S.f16)

    for k_tile in S.range(K_tiles):
        # Load A: load MMA_M x MMA_K tile
        # Each thread loads: A_smem[lane_id/2][lane_id%2 * 8:lane_id%2 * 8 + 8], B_smem[lane_id/2][lane_id%2 * 8:lane_id%2 * 8 + 8] if lane_id < MMA_N*2
        row_a = lane_id >> 1
        col_a = (lane_id & 1) * 8

        row_b = (lane_id % (MMA_N * 2)) >> 1
        col_b = ((lane_id % (MMA_N * 2)) & 1) * 8
        for j in S.range(8):
            a_smem[row_a, col_a + j] = A[warp_row + row_a, k_tile * MMA_K + col_a + j]
            b_smem[row_b, col_b + j] = B[k_tile * MMA_K + col_b + j, warp_col + row_b]

        S.syncthreads()

        # Load from shared memory to registers via ldmatrix
        # Each thread loads a specific 8x8 block from A
        # Thread mapping: lane_id maps to A_smem[lane_id % 16][(lane_id / 16) * 8]
        # This gives us 4 x i32 (8 f16 values) per thread
        a_row_offset = lane_id % 16
        a_col_offset = (lane_id >> 4) * 8
        a_tile = S.subview(a_smem, (a_row_offset, a_col_offset), (8, 8), (1, 1))
        RA = S.nvvm.ldmatrix_m8n8_x4_b16(a_tile)  # 4xi32

        # Similar to A
        b_row_offset = lane_id % 8
        b_col_offset = ((lane_id >> 3) & 1) * 8
        b_tile = S.subview(b_smem, (b_row_offset, b_col_offset), (8, 8), (1, 1))
        RB = S.nvvm.ldmatrix_m8n8_x2_b16(b_tile)  # 2xi32

        # MMA
        RC = S.nvvm.mma_16x8x16_f16_f16(RA, RB, RC)

        S.syncthreads()

    # Store results back to shared memory: each thread writes its 4 f16 values to c_smem
    row1 = lane_id >> 2
    col1 = (lane_id & 3) * 2

    c_smem[row1, col1] = RC[0]
    c_smem[row1, col1 + 1] = RC[1]

    c_smem[row1 + 8, col1] = RC[2]
    c_smem[row1 + 8, col1 + 1] = RC[3]

    S.syncthreads()

    # Each thread writes int4 (128-bit = 8 f16 values) to C
    for j in S.range(8):
        C[warp_row + (lane_id % MMA_M), warp_col + j] = c_smem[(lane_id % MMA_M), j]


@unittest.skipUnless(
    has_cuda_nvidia(),
    "Requires CUDA with an NVIDIA GPU.",
)
class TestGEMM(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        torch.manual_seed(0)
        self.M = 128
        self.N = 128
        self.K = 128
        self.rtol = 1e-1
        self.atol = 1e-1

    def test_gemm_base(self):
        """[NVVM] Test GEMM base with float16 precision using MMA intrinsics."""
        A = torch.randn((self.M, self.K), dtype=torch.float16, device="cuda")
        B = torch.randn((self.K, self.N), dtype=torch.float16, device="cuda")
        C = torch.zeros((self.M, self.N), dtype=torch.float16, device="cuda")

        expected = (A @ B).to(dtype=torch.float16, device="cpu")

        grid_x = (self.N + 7) // 8
        grid_y = (self.M + 15) // 16

        gemm_nvvm_16x8x16[lambda: ((grid_x, grid_y, 1), (32, 1, 1))](A, B, C)

        actual = C.to("cpu")

        self.assertTrue(
            torch.allclose(actual, expected, rtol=self.rtol, atol=self.atol),
            msg=f"GEMM results do not match.\nExpected:\n{expected}\nActual:\n{actual}\n"
            f"Max absolute difference: {torch.max(torch.abs(actual - expected))}",
        )


if __name__ == "__main__":
    unittest.main()
