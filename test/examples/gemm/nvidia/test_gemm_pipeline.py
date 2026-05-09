#!/usr/bin/env python3
import unittest

import torch

import avelang
import avelang.language as S
from avelang.testing import has_cuda_nvidia


@avelang.jit
def gemm_2stages_pipeline(
    A: S.Tensor((128, 128), S.f16),
    B: S.Tensor((128, 128), S.f16),
    C: S.Tensor((128, 128), S.f16),
):
    MMA_M = 16
    MMA_N = 8
    MMA_K = 16

    WARP_SIZE = 32

    # Work group configuration
    WARPS_M = 2  # Number of warps in M dimension per work group
    WARPS_N = 2  # Number of warps in N dimension per work group

    # Each warp computes multiple MMA tiles
    MMA_TILES_M = 2  # Number of MMA tiles per warp in M dimension
    MMA_TILES_N = 2  # Number of MMA tiles per warp in N dimension

    # Block size
    BLOCK_M = WARPS_M * MMA_TILES_M * MMA_M  # Total M covered by work group
    BLOCK_N = WARPS_N * MMA_TILES_N * MMA_N  # Total N covered by work group

    K_tiles = 8  # 128 // 16

    # =========================================== #

    # Work group and warp identification
    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N

    thread_id = S.thread_id(0)
    warp_id = thread_id >> 5  # thread_id // WARP_SIZE
    lane_id = thread_id % WARP_SIZE

    warp_row_id = warp_id >> 1  # warp_id // WARPS_N
    warp_col_id = warp_id % WARPS_N

    # Shared memory: [stage][warp_row][warp_col][local_row][local_col]
    # A: [2][WARPS_M][MMA_TILES_M][MMA_M][MMA_K]
    # B: [2][WARPS_N][MMA_TILES_N][MMA_N][MMA_K]
    a_smem = S.make_shared((2, 2, 2, 16, 16), S.f16)
    b_smem = S.make_shared((2, 2, 2, 8, 16), S.f16)
    c_smem = S.make_shared((2, 2, 2, 2, 16, 8), S.f16)

    # Accumulators for each MMA tile this warp computes
    RC = S.make_local((2, 2, 4), S.f16)

    for tm in S.range(MMA_TILES_M):
        for tn in S.range(MMA_TILES_N):
            for i in S.range(4):
                RC[tm, tn, i] = S.convert(0, S.f16)

    # Precompute addressing for A and B loads
    row_a = lane_id >> 1
    col_a = (lane_id & 1) * 8
    row_b = (lane_id % (MMA_N * 2)) >> 1
    col_b = ((lane_id % (MMA_N * 2)) & 1) * 8

    # Stage 0: Prefetch first tile (k_tile = 0) into buffer 0
    write_stage = 0
    if warp_col_id == 0:
        for tm in S.range(MMA_TILES_M):
            for j in S.range(8):
                global_row = block_row + warp_row_id * (MMA_TILES_M * MMA_M) + tm * MMA_M + row_a
                global_col = col_a + j
                a_smem[write_stage, warp_row_id, tm, row_a, col_a + j] = A[global_row, global_col]

    if warp_row_id == 0:
        for tn in S.range(MMA_TILES_N):
            for j in S.range(8):
                global_row = col_b + j
                global_col = block_col + warp_col_id * (MMA_TILES_N * MMA_N) + tn * MMA_N + row_b
                b_smem[write_stage, warp_col_id, tn, row_b, col_b + j] = B[global_row, global_col]

    S.syncthreads()

    # Main pipeline loop
    for k_tile in S.range(K_tiles - 1):
        read_stage = k_tile & 1
        write_stage = 1 - read_stage

        # Prefetch next tile
        next_k = k_tile + 1
        if warp_col_id == 0:
            for tm in S.range(MMA_TILES_M):
                for j in S.range(8):
                    global_row = block_row + warp_row_id * (MMA_TILES_M * MMA_M) + tm * MMA_M + row_a
                    global_col = next_k * MMA_K + col_a + j
                    a_smem[write_stage, warp_row_id, tm, row_a, col_a + j] = A[global_row, global_col]

        if warp_row_id == 0:
            for tn in S.range(MMA_TILES_N):
                for j in S.range(8):
                    global_row = next_k * MMA_K + col_b + j
                    global_col = block_col + warp_col_id * (MMA_TILES_N * MMA_N) + tn * MMA_N + row_b
                    b_smem[write_stage, warp_col_id, tn, row_b, col_b + j] = B[global_row, global_col]

        # Compute all MMA tiles for this warp
        for tm in S.range(MMA_TILES_M):
            for tn in S.range(MMA_TILES_N):
                # Load A tile
                a_row_offset = lane_id % 16
                a_col_offset = (lane_id >> 4) * 8
                a_smem_tile = S.subview(
                    a_smem,
                    (read_stage, warp_row_id, tm, 0, 0),
                    (1, 1, 1, 16, 16),
                    (1, 1, 1, 1, 1),
                )
                a_tile_2d = S.subview(a_smem_tile, (a_row_offset, a_col_offset), (8, 8), (1, 1))
                RA = S.nvvm.ldmatrix_m8n8_x4_b16(a_tile_2d)

                # Load B tile
                b_row_offset = lane_id % 8
                b_col_offset = ((lane_id >> 3) & 1) * 8
                b_smem_tile = S.subview(
                    b_smem,
                    (read_stage, warp_col_id, tn, 0, 0),
                    (1, 1, 1, 8, 16),
                    (1, 1, 1, 1, 1),
                )
                b_tile_2d = S.subview(b_smem_tile, (b_row_offset, b_col_offset), (8, 8), (1, 1))
                RB = S.nvvm.ldmatrix_m8n8_x2_b16(b_tile_2d)

                # MMA
                RC_view = S.subview(RC, (tm, tn, 0), (1, 1, 4), (1, 1, 1))
                RC_result = S.nvvm.mma_16x8x16_f16_f16(RA, RB, RC_view)

                # Store back to RC
                for i in S.range(4):
                    RC[tm, tn, i] = RC_result[i]

        S.syncthreads()

    # Handle the last k_tile
    last_stage = (K_tiles - 1) & 1

    for tm in S.range(MMA_TILES_M):
        for tn in S.range(MMA_TILES_N):
            # Load A tile
            a_row_offset = lane_id % 16
            a_col_offset = (lane_id >> 4) * 8
            a_smem_tile_last = S.subview(
                a_smem,
                (last_stage, warp_row_id, tm, 0, 0),
                (1, 1, 1, 16, 16),
                (1, 1, 1, 1, 1),
            )
            a_tile_2d_last = S.subview(a_smem_tile_last, (a_row_offset, a_col_offset), (8, 8), (1, 1))
            RA = S.nvvm.ldmatrix_m8n8_x4_b16(a_tile_2d_last)

            # Load B tile
            b_row_offset = lane_id % 8
            b_col_offset = ((lane_id >> 3) & 1) * 8
            b_smem_tile_last = S.subview(
                b_smem,
                (last_stage, warp_col_id, tn, 0, 0),
                (1, 1, 1, 8, 16),
                (1, 1, 1, 1, 1),
            )
            b_tile_2d_last = S.subview(b_smem_tile_last, (b_row_offset, b_col_offset), (8, 8), (1, 1))
            RB = S.nvvm.ldmatrix_m8n8_x2_b16(b_tile_2d_last)

            # MMA
            RC_view = S.subview(RC, (tm, tn, 0), (1, 1, 4), (1, 1, 1))
            RC_result = S.nvvm.mma_16x8x16_f16_f16(RA, RB, RC_view)

            # Store back to RC
            for i in S.range(4):
                RC[tm, tn, i] = RC_result[i]

    S.syncthreads()

    # Store results to shared memory
    row1 = lane_id >> 2
    col1 = (lane_id & 3) * 2

    for tm in S.range(MMA_TILES_M):
        for tn in S.range(MMA_TILES_N):
            c_smem[warp_row_id, warp_col_id, tm, tn, row1, col1] = RC[tm, tn, 0]
            c_smem[warp_row_id, warp_col_id, tm, tn, row1, col1 + 1] = RC[tm, tn, 1]
            c_smem[warp_row_id, warp_col_id, tm, tn, row1 + 8, col1] = RC[tm, tn, 2]
            c_smem[warp_row_id, warp_col_id, tm, tn, row1 + 8, col1 + 1] = RC[tm, tn, 3]

    S.syncthreads()

    # Write results to global memory
    for tm in S.range(MMA_TILES_M):
        for tn in S.range(MMA_TILES_N):
            for j in S.range(MMA_N):
                global_row = block_row + warp_row_id * (MMA_TILES_M * MMA_M) + tm * MMA_M + (lane_id % MMA_M)
                global_col = block_col + warp_col_id * (MMA_TILES_N * MMA_N) + tn * MMA_N + j
                C[global_row, global_col] = c_smem[warp_row_id, warp_col_id, tm, tn, (lane_id % MMA_M), j]


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

    def test_gemm_2stage(self):
        """[NVVM] Test 2-stage GEMM with float16 precision using MMA intrinsics."""
        A = torch.randn((self.M, self.K), dtype=torch.float16, device="cuda")
        B = torch.randn((self.K, self.N), dtype=torch.float16, device="cuda")
        C = torch.zeros((self.M, self.N), dtype=torch.float16, device="cuda")

        expected = (A @ B).to(dtype=torch.float16, device="cpu")

        WARPS_M = 2
        WARPS_N = 2
        MMA_TILES_M = 2
        MMA_TILES_N = 2
        MMA_M = 16
        MMA_N = 8

        BLOCK_M = WARPS_M * MMA_TILES_M * MMA_M  # 2 * 2 * 16 = 64
        BLOCK_N = WARPS_N * MMA_TILES_N * MMA_N  # 2 * 2 * 8 = 32

        grid_x = (self.N + BLOCK_N - 1) // BLOCK_N
        grid_y = (self.M + BLOCK_M - 1) // BLOCK_M

        gemm_2stages_pipeline[lambda: ((grid_x, grid_y, 1), (WARPS_M * WARPS_N * 32, 1, 1))](A, B, C)

        actual = C.to("cpu")

        self.assertTrue(
            torch.allclose(actual, expected, rtol=self.rtol, atol=self.atol),
            msg=f"GEMM results do not match.\nExpected:\n{expected}\nActual:\n{actual}\n"
            f"Max absolute difference: {torch.max(torch.abs(actual - expected))}",
        )


if __name__ == "__main__":
    unittest.main()
