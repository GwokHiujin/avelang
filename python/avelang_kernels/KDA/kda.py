"""Hopper KDA forward kernel for a fixed ``1x8192x64x128`` shape."""

import math

import torch

import avelang
import avelang.language as S


BATCH = 1
SEQUENCE = 8192
HEADS = 64
DIM = 128
CHUNK = 16
TILES = SEQUENCE // CHUNK
OUTER = HEADS * TILES
THREADS = 256
SPLITS = 2
SPLIT_BLOCKS = 4
PREP_STRIDE = 136
STATE_STRIDE = 24
STATE_TILE_ELEMS = CHUNK * STATE_STRIDE
U_STRIDE = 24
U_BLOCK_ELEMS = CHUNK * U_STRIDE
V_STRIDE = 136
LOG2E = 1.4426950408889634
SCALE = 1.0 / math.sqrt(DIM)
GATE_SCALE = -5.0 * LOG2E


@avelang.jit
def _kda_prepare_fixed8192_h64(
    q: S.Tensor((BATCH, SEQUENCE, HEADS, DIM), S.bf16),
    k: S.Tensor((BATCH, SEQUENCE, HEADS, DIM), S.bf16),
    g: S.Tensor((BATCH, SEQUENCE, HEADS, DIM), S.bf16),
    beta: S.Tensor((BATCH, SEQUENCE, HEADS), S.bf16),
    a_log: S.Tensor((HEADS,), S.f32),
    dt_bias: S.Tensor((HEADS, DIM), S.f32),
    kd_ws: S.Tensor((OUTER * 8, 256), S.bf16),
    qd_ws: S.Tensor((OUTER * 8, 256), S.bf16),
    kr_ws: S.Tensor((OUTER * 8, 256), S.bf16),
    gt_ws: S.Tensor((OUTER, DIM), S.bf16),
    inv_ws: S.Tensor((OUTER * CHUNK * CHUNK,), S.bf16),
    mqk_ws: S.Tensor((OUTER * CHUNK * CHUNK,), S.bf16),
    beta_ws: S.Tensor((OUTER * CHUNK,), S.bf16),
):
    tid = S.thread_id(0)
    warp = tid >> 5
    lane = tid & 31
    tile = S.block_id(0)
    head = S.block_id(1)
    t0 = tile * CHUNK
    ws_idx = head * TILES + tile

    # Index zero is kd and index one is qd, allowing the two MMA warps to
    # select their A operand without duplicating the instruction body.
    qk_stage = S.make_shared((2, CHUNK, PREP_STRIDE), S.bf16, 128)
    g_stage = S.make_shared((CHUNK, PREP_STRIDE), S.bf16, 128)
    g_cumsum = S.make_shared((CHUNK, PREP_STRIDE), S.f32, 128)
    gt = S.make_shared((DIM,), S.bf16, 128)
    k_inv = S.make_shared((CHUNK, PREP_STRIDE), S.bf16, 128)
    kr_tma_storage = S.make_shared((8 * 256,), S.bf16, 128)
    kd_tma_storage = S.make_shared((8 * 256,), S.bf16, 128)
    qd_tma_storage = S.make_shared((8 * 256,), S.bf16, 128)
    tma_wide_layout = S.make_layout((8, 256), (256, 1))
    kr_tma_raw = S.view(kr_tma_storage, S.bf16, tma_wide_layout)
    kd_tma_raw = S.view(kd_tma_storage, S.bf16, tma_wide_layout)
    qd_tma_raw = S.view(qd_tma_storage, S.bf16, tma_wide_layout)
    l_tmp = S.make_shared((CHUNK, CHUNK), S.f32, 128)
    mqk = S.make_shared((CHUNK, CHUNK), S.bf16, 128)
    inv = S.make_shared((CHUNK, CHUNK), S.bf16, 128)
    beta_act = S.make_shared((CHUNK,), S.bf16, 128)

    kd_desc = S.nvvm.make_tma_descriptor(kd_ws, S.make_layout((8, 256), (256, 1)))
    qd_desc = S.nvvm.make_tma_descriptor(qd_ws, S.make_layout((8, 256), (256, 1)))
    kr_desc = S.nvvm.make_tma_descriptor(kr_ws, S.make_layout((8, 256), (256, 1)))
    gt_desc = S.nvvm.make_tma_descriptor(gt_ws, S.make_layout((1, DIM), (DIM, 1)))

    row = tid >> 4
    col_vec = tid & 15
    shared_byte = (row * PREP_STRIDE + col_vec * 8) * 2
    global_byte = (((t0 + row) * HEADS + head) * DIM + col_vec * 8) * 2
    S.nvvm.cp_async_cg_shared_global(qk_stage, q, CHUNK * PREP_STRIDE * 2 + shared_byte, global_byte, 16)
    S.nvvm.cp_async_cg_shared_global(qk_stage, k, shared_byte, global_byte, 16)
    S.nvvm.cp_async_cg_shared_global(g_stage, g, shared_byte, global_byte, 16)
    S.nvvm.cp_async_commit_group()
    S.nvvm.cp_async_wait_group(0)
    S.syncthreads()

    gate_col = tid & 127
    segment = tid >> 7
    sum0 = S.convert(0.0, S.f32)
    sum1 = S.convert(0.0, S.f32)
    a_exp = S.nvvm.fast_exp2(a_log[head] * S.convert(LOG2E, S.f32))
    dt = dt_bias[head, gate_col]
    for local_row in S.range(4, unroll=True):
        row0 = segment * 4 + local_row
        row1 = row0 + 8
        gv0 = S.convert(g_stage[row0, gate_col], S.f32) + dt
        gv1 = S.convert(g_stage[row1, gate_col], S.f32) + dt
        gate0 = S.convert(GATE_SCALE, S.f32) * (
            S.nvvm.fast_tanh(S.convert(0.5, S.f32) * a_exp * gv0) * S.convert(0.5, S.f32) + S.convert(0.5, S.f32)
        )
        gate1 = S.convert(GATE_SCALE, S.f32) * (
            S.nvvm.fast_tanh(S.convert(0.5, S.f32) * a_exp * gv1) * S.convert(0.5, S.f32) + S.convert(0.5, S.f32)
        )
        sum0 = sum0 + gate0
        sum1 = sum1 + gate1
        g_cumsum[row0, gate_col] = sum0
        g_cumsum[row1, gate_col] = sum1
    S.syncthreads()

    if tid < DIM:
        prefix = g_cumsum[3, gate_col]
        for gate_segment in S.range(1, 4, unroll=True):
            row_begin = gate_segment * 4
            segment_sum = g_cumsum[row_begin + 3, gate_col]
            for local_row in S.range(4, unroll=True):
                current_row = row_begin + local_row
                g_cumsum[current_row, gate_col] = g_cumsum[current_row, gate_col] + prefix
            prefix = prefix + segment_sum
        gt[gate_col] = S.convert(S.nvvm.fast_exp2(prefix), S.bf16)
    S.syncthreads()

    for pair_iteration in S.range(4, unroll=True):
        pair = tid + pair_iteration * THREADS
        element = pair << 1
        transform_row = element >> 7
        transform_col = element & 127
        gc0 = g_cumsum[transform_row, transform_col]
        gc1 = g_cumsum[transform_row, transform_col + 1]
        exp0 = S.nvvm.fast_exp2(gc0)
        exp1 = S.nvvm.fast_exp2(gc1)
        inv0 = S.nvvm.fast_rcp(exp0)
        inv1 = S.nvvm.fast_rcp(exp1)
        q0 = S.convert(qk_stage[1, transform_row, transform_col], S.f32)
        q1 = S.convert(qk_stage[1, transform_row, transform_col + 1], S.f32)
        k0 = S.convert(qk_stage[0, transform_row, transform_col], S.f32)
        k1 = S.convert(qk_stage[0, transform_row, transform_col + 1], S.f32)
        qk_stage[1, transform_row, transform_col] = S.convert(q0 * exp0 * S.convert(SCALE, S.f32), S.bf16)
        qk_stage[1, transform_row, transform_col + 1] = S.convert(q1 * exp1 * S.convert(SCALE, S.f32), S.bf16)
        qk_stage[0, transform_row, transform_col] = S.convert(k0 * exp0, S.bf16)
        qk_stage[0, transform_row, transform_col + 1] = S.convert(k1 * exp1, S.bf16)
        k_inv[transform_row, transform_col] = S.convert(k0 * inv0, S.bf16)
        k_inv[transform_row, transform_col + 1] = S.convert(k1 * inv1, S.bf16)
        kr_offset0 = (transform_col >> 4) * 256 + transform_row * 16 + (transform_col & 15)
        kr_offset1 = kr_offset0 + 1
        kr_tma_raw[kr_offset0 // 256, kr_offset0 & 255] = S.convert(
            k0 * inv0 * S.convert(gt[transform_col], S.f32), S.bf16
        )
        kr_tma_raw[kr_offset1 // 256, kr_offset1 & 255] = S.convert(
            k1 * inv1 * S.convert(gt[transform_col + 1], S.f32), S.bf16
        )
    S.syncthreads()

    if tid == 0:
        S.nvvm.fence_proxy_async_shared_cta()
        S.nvvm.tma_store(kr_tma_raw, kr_desc, (0, ws_idx * 8))
        gt_tma = S.view(gt, S.bf16, S.make_layout((1, DIM), (DIM, 1)))
        S.nvvm.tma_store(gt_tma, gt_desc, (0, ws_idx))
        S.nvvm.cp_async_bulk_commit_group()

    # TMA-wide kd/qd are stored in the ldmatrix-native interleaved layout.
    inter_row = tid & 15
    inter_col = (tid >> 4) * 8
    for element_in_vector in S.range(8, unroll=True):
        kd_tma_raw[(tid * 8 + element_in_vector) // 256, (tid * 8 + element_in_vector) & 255] = qk_stage[
            0, inter_row, inter_col + element_in_vector
        ]
        qd_tma_raw[(tid * 8 + element_in_vector) // 256, (tid * 8 + element_in_vector) & 255] = qk_stage[
            1, inter_row, inter_col + element_in_vector
        ]
    S.syncthreads()
    if tid == 0:
        S.nvvm.fence_proxy_async_shared_cta()
        S.nvvm.tma_store(kd_tma_raw, kd_desc, (0, ws_idx * 8))
        S.nvvm.tma_store(qd_tma_raw, qd_desc, (0, ws_idx * 8))
        S.nvvm.cp_async_bulk_commit_group()

    if tid >= 64 and tid < 80:
        beta_row = tid - 64
        beta_value = S.convert(beta[0, t0 + beta_row, head], S.f32)
        sigmoid = S.nvvm.fast_tanh(S.convert(0.5, S.f32) * beta_value) * S.convert(0.5, S.f32) + S.convert(0.5, S.f32)
        beta_act[beta_row] = S.convert(sigmoid, S.bf16)
        beta_ws[ws_idx * CHUNK + beta_row] = S.convert(sigmoid, S.bf16)

    if warp < 2:
        c0 = S.full((4,), 0.0, S.f32)
        c1 = S.full((4,), 0.0, S.f32)
        for k_block in S.range(8, unroll=True):
            matrix_row = (((lane >> 3) & 1) << 3) + (lane & 7)
            matrix_col = (lane >> 4) << 3
            a_tile = S.subview(
                qk_stage,
                (warp, matrix_row, k_block * 16 + matrix_col),
                (1, 8, 8),
                (1, 1, 1),
            )
            b_tile = S.subview(
                k_inv,
                (matrix_row, k_block * 16 + matrix_col),
                (8, 8),
                (1, 1),
            )
            a_frag = S.nvvm.ldmatrix_m8n8_x4_b16(a_tile)
            b_frag = S.nvvm.ldmatrix_m8n8_x4_b16(b_tile)
            b_lo = S.full((2,), 0, S.i32)
            b_hi = S.full((2,), 0, S.i32)
            b_lo[0] = b_frag[0]
            b_lo[1] = b_frag[2]
            b_hi[0] = b_frag[1]
            b_hi[1] = b_frag[3]
            c0 = S.nvvm.mma_16x8x16_bf16_f32(a_frag, b_lo, c0)
            c1 = S.nvvm.mma_16x8x16_bf16_f32(a_frag, b_hi, c1)

        result_row = lane >> 2
        result_col = (lane & 3) << 1
        if warp == 0:
            l_tmp[result_row, result_col] = c0[0]
            l_tmp[result_row, result_col + 1] = c0[1]
            l_tmp[result_row + 8, result_col] = c0[2]
            l_tmp[result_row + 8, result_col + 1] = c0[3]
            l_tmp[result_row, result_col + 8] = c1[0]
            l_tmp[result_row, result_col + 9] = c1[1]
            l_tmp[result_row + 8, result_col + 8] = c1[2]
            l_tmp[result_row + 8, result_col + 9] = c1[3]
        else:
            mqk[result_row, result_col] = S.convert(
                S.select(result_row < result_col, S.convert(0.0, S.f32), c0[0]), S.bf16
            )
            mqk[result_row, result_col + 1] = S.convert(
                S.select(result_row < result_col + 1, S.convert(0.0, S.f32), c0[1]), S.bf16
            )
            mqk[result_row + 8, result_col] = S.convert(
                S.select(result_row + 8 < result_col, S.convert(0.0, S.f32), c0[2]), S.bf16
            )
            mqk[result_row + 8, result_col + 1] = S.convert(
                S.select(result_row + 8 < result_col + 1, S.convert(0.0, S.f32), c0[3]), S.bf16
            )
            mqk[result_row, result_col + 8] = S.convert(
                S.select(result_row < result_col + 8, S.convert(0.0, S.f32), c1[0]), S.bf16
            )
            mqk[result_row, result_col + 9] = S.convert(
                S.select(result_row < result_col + 9, S.convert(0.0, S.f32), c1[1]), S.bf16
            )
            mqk[result_row + 8, result_col + 8] = S.convert(
                S.select(result_row + 8 < result_col + 8, S.convert(0.0, S.f32), c1[2]), S.bf16
            )
            mqk[result_row + 8, result_col + 9] = S.convert(
                S.select(result_row + 8 < result_col + 9, S.convert(0.0, S.f32), c1[3]), S.bf16
            )
    S.syncthreads()

    if tid < 128:
        element = tid << 1
        inv_row = element >> 4
        inv_col = element & 15
        beta_value = S.convert(beta_act[inv_row], S.f32)
        lower0 = S.select(
            inv_row > inv_col,
            l_tmp[inv_row, inv_col] * beta_value,
            S.convert(0.0, S.f32),
        )
        lower1 = S.select(
            inv_row > inv_col + 1,
            l_tmp[inv_row, inv_col + 1] * beta_value,
            S.convert(0.0, S.f32),
        )
        inv[inv_row, inv_col] = S.convert(
            S.select(
                inv_row == inv_col,
                S.convert(1.0, S.f32),
                S.convert(0.0, S.f32),
            )
            - lower0,
            S.bf16,
        )
        inv[inv_row, inv_col + 1] = S.convert(
            S.select(
                inv_row == inv_col + 1,
                S.convert(1.0, S.f32),
                S.convert(0.0, S.f32),
            )
            - lower1,
            S.bf16,
        )
        inter0 = (inv_row & 7) * 8 + (inv_row >> 3) * 64 + (inv_col & 7) + (inv_col >> 3) * 128
        inter1 = (inv_row & 7) * 8 + (inv_row >> 3) * 64 + ((inv_col + 1) & 7) + ((inv_col + 1) >> 3) * 128
        inv_ws[ws_idx * 256 + inter0] = inv[inv_row, inv_col]
        inv_ws[ws_idx * 256 + inter1] = inv[inv_row, inv_col + 1]
        mqk_ws[ws_idx * 256 + inter0] = mqk[inv_row, inv_col]
        mqk_ws[ws_idx * 256 + inter1] = mqk[inv_row, inv_col + 1]


@avelang.jit
def _kda_recurrence_fixed8192_h64_split2(
    v: S.Tensor((BATCH, SEQUENCE, HEADS, DIM), S.bf16),
    out: S.Tensor((BATCH, SEQUENCE, HEADS, DIM), S.bf16),
    kd_ws: S.Tensor((OUTER * 8, 256), S.bf16),
    qd_ws: S.Tensor((OUTER * 8, 256), S.bf16),
    kr_ws: S.Tensor((OUTER * 8, 256), S.bf16),
    gt_ws: S.Tensor((OUTER, DIM), S.bf16),
    inv_ws: S.Tensor((OUTER * CHUNK * CHUNK,), S.bf16),
    mqk_ws: S.Tensor((OUTER * CHUNK * CHUNK,), S.bf16),
    beta_ws: S.Tensor((OUTER * CHUNK,), S.bf16),
    sync_mask_lo: S.i32,
):
    tid = S.thread_id(0)
    warp = tid >> 5
    lane = tid & 31
    head = S.block_id(1)
    split = S.block_id(2)
    row_block_begin = split * SPLIT_BLOCKS

    # LLVM emits workgroup globals in reverse declaration order. Declare the
    # arena in reverse to retain the intended packed shared-memory offsets.
    u_tile = S.make_shared((8, CHUNK, U_STRIDE), S.bf16, 16)
    gt_stage = S.make_shared((2, DIM), S.bf16, 16)
    beta_stage = S.make_shared((2, CHUNK), S.bf16, 16)
    mqk_stage = S.make_shared((2, CHUNK, CHUNK), S.bf16, 16)
    inv_stage = S.make_shared((2, CHUNK, CHUNK), S.bf16, 16)
    v_stage = S.make_shared((2, CHUNK, V_STRIDE), S.bf16, 16)
    kr_stage = S.make_shared((2, CHUNK, DIM), S.bf16, 16)
    qd_stage = S.make_shared((2, CHUNK, DIM), S.bf16, 16)
    kd_stage = S.make_shared((2, CHUNK, DIM), S.bf16, 16)
    state = S.make_shared((SPLIT_BLOCKS * 8, CHUNK, STATE_STRIDE), S.bf16, 16)
    state_smem = S.nvvm.shared_address(state)
    kd_smem = S.nvvm.shared_address(kd_stage)
    qd_smem = S.nvvm.shared_address(qd_stage)
    kr_smem = S.nvvm.shared_address(kr_stage)
    v_smem = S.nvvm.shared_address(v_stage)
    inv_smem = S.nvvm.shared_address(inv_stage)
    mqk_smem = S.nvvm.shared_address(mqk_stage)
    u_smem = S.nvvm.shared_address(u_tile)
    state_u32 = S.view(
        state,
        S.u32,
        S.make_layout((SPLIT_BLOCKS * 8 * STATE_TILE_ELEMS // 2,), (1,)),
    )
    u_u32 = S.view(u_tile, S.u32, S.make_layout((8 * U_BLOCK_ELEMS // 2,), (1,)))
    out_u32 = S.view(
        out,
        S.u32,
        S.make_layout((BATCH * SEQUENCE * HEADS * DIM // 2,), (1,)),
    )

    # No state input in the selected specialization.
    for pair_iteration in S.range(16, unroll=True):
        pair = tid + pair_iteration * THREADS
        local_row = pair >> 6
        state_col = (pair & 63) << 1
        state_offset = (
            ((local_row >> 4) * 8 + (state_col >> 4)) * STATE_TILE_ELEMS
            + (local_row & 15) * STATE_STRIDE
            + (state_col & 15)
        )
        state_u32[state_offset >> 1] = 0
    S.syncthreads()

    # Retain the dynamic BF16-state load path in the binary; split is always
    # 0 or 1 for this launch, so this branch is inactive at runtime.
    if split >= SPLITS:
        for retained_copy in S.range(5, unroll=True):
            S.nvvm.cp_async_cg_shared_global(
                state,
                v,
                retained_copy * 16 + tid * 0,
                retained_copy * 16 + tid * 0,
                16,
            )
        S.nvvm.cp_async_commit_group()
        S.nvvm.cp_async_wait_group(0)

    # Prime the two-stage asynchronous-copy pipeline.
    ws_idx = head * TILES
    workspace_byte = ws_idx * CHUNK * DIM * 2
    S.nvvm.cp_async_cg_shared_global(kd_stage, kd_ws, tid * 16, workspace_byte + tid * 16, 16)
    S.nvvm.cp_async_cg_shared_global(qd_stage, qd_ws, tid * 16, workspace_byte + tid * 16, 16)
    S.nvvm.cp_async_cg_shared_global(kr_stage, kr_ws, tid * 16, workspace_byte + tid * 16, 16)
    if tid < 128:
        v_row = tid >> 3
        v_local_vec = tid & 7
        v_col_vec = split * 8 + v_local_vec
        S.nvvm.cp_async_cg_shared_global(
            v_stage,
            v,
            (v_row * V_STRIDE + v_col_vec * 8) * 2,
            ((v_row * HEADS + head) * DIM + v_col_vec * 8) * 2,
            16,
        )
    if tid < 32:
        S.nvvm.cp_async_cg_shared_global(inv_stage, inv_ws, tid * 16, (ws_idx * 256 * 2) + tid * 16, 16)
        S.nvvm.cp_async_cg_shared_global(mqk_stage, mqk_ws, tid * 16, (ws_idx * 256 * 2) + tid * 16, 16)
    if tid < 16:
        S.nvvm.cp_async_cg_shared_global(gt_stage, gt_ws, tid * 16, (ws_idx * DIM * 2) + tid * 16, 16)
    if tid < 2:
        S.nvvm.cp_async_cg_shared_global(beta_stage, beta_ws, tid * 16, (ws_idx * CHUNK * 2) + tid * 16, 16)
    S.nvvm.cp_async_commit_group()
    S.nvvm.cp_async_wait_group(0)
    S.syncthreads()

    for tile in S.range(TILES):
        stage = tile & 1
        t0 = tile * CHUNK
        ws_idx = head * TILES + tile

        if tile + 1 < TILES:
            next_tile = tile + 1
            next_stage = next_tile & 1
            next_ws = head * TILES + next_tile
            next_tile_offset = next_stage * CHUNK * DIM
            next_workspace_byte = next_ws * CHUNK * DIM * 2
            S.nvvm.cp_async_cg_shared_global(
                kd_stage,
                kd_ws,
                next_tile_offset * 2 + tid * 16,
                next_workspace_byte + tid * 16,
                16,
            )
            S.nvvm.cp_async_cg_shared_global(
                qd_stage,
                qd_ws,
                next_tile_offset * 2 + tid * 16,
                next_workspace_byte + tid * 16,
                16,
            )
            S.nvvm.cp_async_cg_shared_global(
                kr_stage,
                kr_ws,
                next_tile_offset * 2 + tid * 16,
                next_workspace_byte + tid * 16,
                16,
            )
            if tid < 128:
                v_row = tid >> 3
                v_local_vec = tid & 7
                v_col_vec = split * 8 + v_local_vec
                S.nvvm.cp_async_cg_shared_global(
                    v_stage,
                    v,
                    (next_stage * CHUNK * V_STRIDE + v_row * V_STRIDE + v_col_vec * 8) * 2,
                    (((next_tile * CHUNK + v_row) * HEADS + head) * DIM + v_col_vec * 8) * 2,
                    16,
                )
            if tid < 32:
                S.nvvm.cp_async_cg_shared_global(
                    inv_stage,
                    inv_ws,
                    (next_stage * 256) * 2 + tid * 16,
                    (next_ws * 256) * 2 + tid * 16,
                    16,
                )
                S.nvvm.cp_async_cg_shared_global(
                    mqk_stage,
                    mqk_ws,
                    (next_stage * 256) * 2 + tid * 16,
                    (next_ws * 256) * 2 + tid * 16,
                    16,
                )
            if tid < 16:
                S.nvvm.cp_async_cg_shared_global(
                    gt_stage,
                    gt_ws,
                    (next_stage * DIM) * 2 + tid * 16,
                    (next_ws * DIM) * 2 + tid * 16,
                    16,
                )
            if tid < 2:
                S.nvvm.cp_async_cg_shared_global(
                    beta_stage,
                    beta_ws,
                    (next_stage * CHUNK) * 2 + tid * 16,
                    (next_ws * CHUNK) * 2 + tid * 16,
                    16,
                )
            S.nvvm.cp_async_commit_group()

        if warp < SPLIT_BLOCKS:
            k_c0 = S.full((4,), 0.0, S.f32)
            k_c1 = S.full((4,), 0.0, S.f32)
            q_c0 = S.full((4,), 0.0, S.f32)
            q_c1 = S.full((4,), 0.0, S.f32)
            matrix_row = (((lane >> 3) & 1) << 3) + (lane & 7)
            matrix_col = (lane >> 4) << 3
            # Preserve the half-warp synchronization edges. The stores touch
            # only the eight-column state padding and keep ptxas
            # from folding the divergent warp barriers into a single barrier.
            if lane < 16:
                for padding_row in S.range(4, unroll=True):
                    padding_offset = (
                        warp * STATE_TILE_ELEMS + ((lane >> 2) + padding_row * 4) * STATE_STRIDE + 16 + (lane & 3) * 2
                    )
                    state_u32[padding_offset >> 1] = 0
                S.nvvm.syncwarp(sync_mask_lo)
            for k_block in S.range(8, unroll=True):
                state_block = warp * 8 + k_block
                state_address = (
                    state_smem + (state_block * STATE_TILE_ELEMS + matrix_row * STATE_STRIDE + matrix_col) * 2
                )
                b_state = S.nvvm.ldmatrix_m8n8_x4_b16(state_address)

                inter_address = (lane & 7) * 8 + ((lane >> 3) & 1) * 64 + (k_block * 2 + (lane >> 4)) * 128
                a_k = S.nvvm.ldmatrix_m8n8_x4_b16(kd_smem + (stage * CHUNK * DIM + inter_address) * 2)
                a_q = S.nvvm.ldmatrix_m8n8_x4_b16(qd_smem + (stage * CHUNK * DIM + inter_address) * 2)
                b_lo = S.full((2,), 0, S.i32)
                b_hi = S.full((2,), 0, S.i32)
                b_lo[0] = b_state[0]
                b_lo[1] = b_state[2]
                b_hi[0] = b_state[1]
                b_hi[1] = b_state[3]
                k_c0 = S.nvvm.mma_16x8x16_bf16_f32(a_k, b_lo, k_c0)
                k_c1 = S.nvvm.mma_16x8x16_bf16_f32(a_k, b_hi, k_c1)
                q_c0 = S.nvvm.mma_16x8x16_bf16_f32(a_q, b_lo, q_c0)
                q_c1 = S.nvvm.mma_16x8x16_bf16_f32(a_q, b_hi, q_c1)

            col_block = row_block_begin + warp
            col0 = col_block * 16
            result_row = lane >> 2
            result_col = (lane & 3) << 1
            S.nvvm.syncwarp()
            v_address = v_smem + (stage * CHUNK * V_STRIDE + matrix_row * V_STRIDE + col0 + matrix_col) * 2
            v_frag = S.nvvm.ldmatrix_m8n8_x4_b16(v_address)
            S.nvvm.syncwarp()
            beta_top = S.convert(beta_stage[stage, result_row], S.f32)
            beta_bottom = S.convert(beta_stage[stage, result_row + 8], S.f32)
            beta_top_pair = S.nvvm.floatx2_to_bf16x2(beta_top, beta_top)
            beta_bottom_pair = S.nvvm.floatx2_to_bf16x2(beta_bottom, beta_bottom)
            u_base = col_block * U_BLOCK_ELEMS
            u_top = result_row * U_STRIDE + result_col
            u_bottom = (result_row + 8) * U_STRIDE + result_col
            k_pair0 = S.nvvm.floatx2_to_bf16x2(k_c0[0], k_c0[1])
            k_pair1 = S.nvvm.floatx2_to_bf16x2(k_c0[2], k_c0[3])
            k_pair2 = S.nvvm.floatx2_to_bf16x2(k_c1[0], k_c1[1])
            k_pair3 = S.nvvm.floatx2_to_bf16x2(k_c1[2], k_c1[3])
            u_pair0 = S.nvvm.bf16x2_mul(S.nvvm.bf16x2_sub(v_frag[0], k_pair0), beta_top_pair)
            u_pair1 = S.nvvm.bf16x2_mul(S.nvvm.bf16x2_sub(v_frag[1], k_pair1), beta_bottom_pair)
            u_pair2 = S.nvvm.bf16x2_mul(S.nvvm.bf16x2_sub(v_frag[2], k_pair2), beta_top_pair)
            u_pair3 = S.nvvm.bf16x2_mul(S.nvvm.bf16x2_sub(v_frag[3], k_pair3), beta_bottom_pair)
            u_u32[(u_base + u_top) >> 1] = u_pair0
            u_u32[(u_base + u_bottom) >> 1] = u_pair1
            u_u32[(u_base + u_top + 8) >> 1] = u_pair2
            u_u32[(u_base + u_bottom + 8) >> 1] = u_pair3
            # Preserve the second half-warp synchronization edge.
            if lane < 16:
                for padding_row in S.range(4, unroll=True):
                    padding_offset = (
                        warp * STATE_TILE_ELEMS + ((lane >> 2) + padding_row * 4) * STATE_STRIDE + 16 + (lane & 3) * 2
                    )
                    state_u32[padding_offset >> 1] = 0
                S.nvvm.syncwarp(sync_mask_lo)
            S.nvvm.syncwarp()

            inv_address = (matrix_row & 7) * 8 + (matrix_row >> 3) * 64 + (matrix_col & 7) + (matrix_col >> 3) * 128
            inv_a = S.nvvm.ldmatrix_m8n8_x4_b16(inv_smem + (stage * 256 + inv_address) * 2)
            S.nvvm.syncwarp()
            u_address = u_smem + (col_block * U_BLOCK_ELEMS + matrix_row * U_STRIDE + matrix_col) * 2
            u_b = S.nvvm.ldmatrix_m8n8_x4_b16_trans(u_address)
            S.nvvm.syncwarp()
            u_b_lo = S.full((2,), 0, S.i32)
            u_b_hi = S.full((2,), 0, S.i32)
            u_b_lo[0] = u_b[0]
            u_b_lo[1] = u_b[1]
            u_b_hi[0] = u_b[2]
            u_b_hi[1] = u_b[3]
            inv_c0 = S.full((4,), 0.0, S.f32)
            inv_c1 = S.full((4,), 0.0, S.f32)
            inv_c0 = S.nvvm.mma_16x8x16_bf16_f32(inv_a, u_b_lo, inv_c0)
            inv_c1 = S.nvvm.mma_16x8x16_bf16_f32(inv_a, u_b_hi, inv_c1)
            u_u32[(u_base + u_top) >> 1] = S.nvvm.floatx2_to_bf16x2(inv_c0[0], inv_c0[1])
            u_u32[(u_base + u_bottom) >> 1] = S.nvvm.floatx2_to_bf16x2(inv_c0[2], inv_c0[3])
            u_u32[(u_base + u_top + 8) >> 1] = S.nvvm.floatx2_to_bf16x2(inv_c1[0], inv_c1[1])
            u_u32[(u_base + u_bottom + 8) >> 1] = S.nvvm.floatx2_to_bf16x2(inv_c1[2], inv_c1[3])
            S.nvvm.syncwarp()

            mqk_address = (matrix_row & 7) * 8 + (matrix_row >> 3) * 64 + (matrix_col & 7) + (matrix_col >> 3) * 128
            mqk_a = S.nvvm.ldmatrix_m8n8_x4_b16(mqk_smem + (stage * 256 + mqk_address) * 2)
            S.nvvm.syncwarp()
            u_b = S.nvvm.ldmatrix_m8n8_x4_b16_trans(u_address)
            u_b_lo[0] = u_b[0]
            u_b_lo[1] = u_b[1]
            u_b_hi[0] = u_b[2]
            u_b_hi[1] = u_b[3]
            out_c0 = S.full((4,), 0.0, S.f32)
            out_c1 = S.full((4,), 0.0, S.f32)
            out_c0 = S.nvvm.mma_16x8x16_bf16_f32(mqk_a, u_b_lo, out_c0)
            out_c1 = S.nvvm.mma_16x8x16_bf16_f32(mqk_a, u_b_hi, out_c1)

            q_pair0 = S.nvvm.floatx2_to_bf16x2(q_c0[0], q_c0[1])
            q_pair1 = S.nvvm.floatx2_to_bf16x2(q_c0[2], q_c0[3])
            q_pair2 = S.nvvm.floatx2_to_bf16x2(q_c1[0], q_c1[1])
            q_pair3 = S.nvvm.floatx2_to_bf16x2(q_c1[2], q_c1[3])
            out_pair0 = S.nvvm.floatx2_to_bf16x2(out_c0[0], out_c0[1])
            out_pair1 = S.nvvm.floatx2_to_bf16x2(out_c0[2], out_c0[3])
            out_pair2 = S.nvvm.floatx2_to_bf16x2(out_c1[0], out_c1[1])
            out_pair3 = S.nvvm.floatx2_to_bf16x2(out_c1[2], out_c1[3])
            output_base = ((t0 + result_row) * HEADS + head) * DIM
            output_bottom = ((t0 + result_row + 8) * HEADS + head) * DIM
            out_u32[(output_base + col0 + result_col) >> 1] = S.nvvm.bf16x2_add(q_pair0, out_pair0)
            out_u32[(output_bottom + col0 + result_col) >> 1] = S.nvvm.bf16x2_add(q_pair1, out_pair1)
            out_u32[(output_base + col0 + result_col + 8) >> 1] = S.nvvm.bf16x2_add(q_pair2, out_pair2)
            out_u32[(output_bottom + col0 + result_col + 8) >> 1] = S.nvvm.bf16x2_add(q_pair3, out_pair3)

        # Preload the column-wise update operand before the CTA barrier.
        matrix_row = (((lane >> 3) & 1) << 3) + (lane & 7)
        matrix_col = (lane >> 4) << 3
        kr_address = warp * 256 + matrix_row * 16 + matrix_col
        update_b = S.nvvm.ldmatrix_m8n8_x4_b16_trans(kr_smem + (stage * CHUNK * DIM + kr_address) * 2)
        result_col = (lane & 3) << 1
        gt_col = warp * 16 + result_col
        gt0 = gt_stage[stage, gt_col]
        gt1 = gt_stage[stage, gt_col + 1]
        gt2 = gt_stage[stage, gt_col + 8]
        gt3 = gt_stage[stage, gt_col + 9]
        S.syncthreads()

        b_lo = S.full((2,), 0, S.i32)
        b_hi = S.full((2,), 0, S.i32)
        b_lo[0] = update_b[0]
        b_lo[1] = update_b[1]
        b_hi[0] = update_b[2]
        b_hi[1] = update_b[3]
        for local_row_block in S.range(SPLIT_BLOCKS, unroll=True):
            logical_row_block = row_block_begin + local_row_block
            u_base = logical_row_block * U_BLOCK_ELEMS
            update_a = S.nvvm.ldmatrix_m8n8_x4_b16_trans(
                u_smem + (logical_row_block * U_BLOCK_ELEMS + matrix_row * U_STRIDE + matrix_col) * 2
            )
            state_block = local_row_block * 8 + warp
            state_base = state_block * STATE_TILE_ELEMS
            state_frag = S.nvvm.ldmatrix_m8n8_x4_b16(
                state_smem + (state_block * STATE_TILE_ELEMS + matrix_row * STATE_STRIDE + matrix_col) * 2
            )
            update_c0 = S.full((4,), 0.0, S.f32)
            update_c1 = S.full((4,), 0.0, S.f32)
            update_c0 = S.nvvm.mma_16x8x16_bf16_f32(update_a, b_lo, update_c0)
            update_c1 = S.nvvm.mma_16x8x16_bf16_f32(update_a, b_hi, update_c1)
            state_row = lane >> 2
            state_col = (lane & 3) << 1
            state_top = state_base + state_row * STATE_STRIDE + state_col
            state_bottom = state_base + (state_row + 8) * STATE_STRIDE + state_col
            gt_pair0 = S.nvvm.floatx2_to_bf16x2(S.convert(gt0, S.f32), S.convert(gt1, S.f32))
            gt_pair1 = S.nvvm.floatx2_to_bf16x2(S.convert(gt2, S.f32), S.convert(gt3, S.f32))
            update_pair0 = S.nvvm.floatx2_to_bf16x2(update_c0[0], update_c0[1])
            update_pair1 = S.nvvm.floatx2_to_bf16x2(update_c0[2], update_c0[3])
            update_pair2 = S.nvvm.floatx2_to_bf16x2(update_c1[0], update_c1[1])
            update_pair3 = S.nvvm.floatx2_to_bf16x2(update_c1[2], update_c1[3])
            state_u32[state_top >> 1] = S.nvvm.bf16x2_fma(state_frag[0], gt_pair0, update_pair0)
            state_u32[state_bottom >> 1] = S.nvvm.bf16x2_fma(state_frag[1], gt_pair0, update_pair1)
            state_u32[(state_top + 8) >> 1] = S.nvvm.bf16x2_fma(state_frag[2], gt_pair1, update_pair2)
            state_u32[(state_bottom + 8) >> 1] = S.nvvm.bf16x2_fma(state_frag[3], gt_pair1, update_pair3)

        if tile + 1 < TILES:
            S.nvvm.cp_async_wait_group(0)
            S.syncthreads()

    # Preserve the fifth static CTA barrier in the no-final-state specialization.
    S.syncthreads()


def get_workspace_size() -> int:
    """Return the selected specialization's workspace size in bytes."""

    elements = 3 * OUTER * CHUNK * DIM + OUTER * DIM + 2 * OUTER * CHUNK * CHUNK + OUTER * CHUNK
    return elements * 2


def _workspace_views(workspace: torch.Tensor):
    offset = 0

    def take(elements, shape):
        nonlocal offset
        result = workspace.narrow(0, offset, elements).view(shape)
        offset += elements
        return result

    tile_elements = OUTER * CHUNK * DIM
    kd = take(tile_elements, (OUTER * 8, 256))
    qd = take(tile_elements, (OUTER * 8, 256))
    kr = take(tile_elements, (OUTER * 8, 256))
    gt = take(OUTER * DIM, (OUTER, DIM))
    inv = take(OUTER * CHUNK * CHUNK, (OUTER * CHUNK * CHUNK,))
    mqk = take(OUTER * CHUNK * CHUNK, (OUTER * CHUNK * CHUNK,))
    beta = take(OUTER * CHUNK, (OUTER * CHUNK,))
    return kd, qd, kr, gt, inv, mqk, beta


def fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    out: torch.Tensor,
    *,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    initial_state=None,
    final_state=None,
    cu_seqlens=None,
):
    """Run fixed ``1x8192x64x128`` KDA without state or variable lengths."""

    expected = (BATCH, SEQUENCE, HEADS, DIM)
    for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g), ("out", out)):
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}")
        if tensor.dtype != torch.bfloat16 or tensor.device.type != "cuda":
            raise ValueError(f"{name} must be contiguous CUDA BF16")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if tuple(beta.shape) != (BATCH, SEQUENCE, HEADS):
        raise ValueError("beta must have shape (1, 8192, 64)")
    if beta.dtype != torch.bfloat16 or not beta.is_cuda or not beta.is_contiguous():
        raise ValueError("beta must be contiguous CUDA BF16")
    if tuple(A_log.shape) != (HEADS,) or A_log.dtype != torch.float32 or not A_log.is_cuda or not A_log.is_contiguous():
        raise ValueError("A_log must be CUDA FP32 with shape (64,)")
    if (
        tuple(dt_bias.shape) != (HEADS, DIM)
        or dt_bias.dtype != torch.float32
        or not dt_bias.is_cuda
        or not dt_bias.is_contiguous()
    ):
        raise ValueError("dt_bias must be CUDA FP32 with shape (64, 128)")
    if initial_state is not None or final_state is not None or cu_seqlens is not None:
        raise ValueError("the selected fixed specialization does not accept state/varlen")
    if not math.isclose(float(scale), SCALE, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"scale must be 1/sqrt(128), got {scale}")
    if float(lower_bound) != -5.0:
        raise ValueError("lower_bound must be -5.0")

    workspace_elements = get_workspace_size() // 2
    workspace = torch.empty(workspace_elements, dtype=torch.bfloat16, device=q.device)
    kd, qd, kr, gt, inv, mqk, beta_act = _workspace_views(workspace)

    _kda_prepare_fixed8192_h64[lambda: ((TILES, HEADS, 1), (THREADS, 1, 1))](
        q,
        k,
        g,
        beta,
        A_log,
        dt_bias,
        kd,
        qd,
        kr,
        gt,
        inv,
        mqk,
        beta_act,
        num_warps=8,
        min_ctas=2,
        prefer_l1=False,
    )
    _kda_recurrence_fixed8192_h64_split2[lambda: ((1, HEADS, SPLITS), (THREADS, 1, 1))](
        v,
        out,
        kd,
        qd,
        kr,
        gt,
        inv,
        mqk,
        beta_act,
        0x0000FFFF,
        num_warps=8,
        min_ctas=2,
        prefer_l1=False,
    )
    return out
