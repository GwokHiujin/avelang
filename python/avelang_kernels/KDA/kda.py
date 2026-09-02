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
VARLEN_SEQ_LENS = (1300, 547, 2048, 963, 271, 3063)
VARLEN_CU_SEQLENS = (0, 1300, 1847, 3895, 4858, 5129, 8192)
VARLEN_TILE_BASES = (0, 82, 117, 245, 306, 323)
VARLEN_TILE_COUNTS = (82, 35, 128, 61, 17, 192)
VARLEN_TILES = 515
VARLEN_OUTER = HEADS * VARLEN_TILES
VARLEN_SEQUENCES = len(VARLEN_SEQ_LENS)
_verified_varlen_metadata = set()


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
def _kda_prepare_varlen_mix6_h64(
    q: S.Tensor((BATCH, SEQUENCE, HEADS, DIM), S.bf16),
    k: S.Tensor((BATCH, SEQUENCE, HEADS, DIM), S.bf16),
    g: S.Tensor((BATCH, SEQUENCE, HEADS, DIM), S.bf16),
    beta: S.Tensor((BATCH, SEQUENCE, HEADS), S.bf16),
    a_log: S.Tensor((HEADS,), S.f32),
    dt_bias: S.Tensor((HEADS, DIM), S.f32),
    kd_ws: S.Tensor((VARLEN_OUTER * 8, 256), S.bf16),
    qd_ws: S.Tensor((VARLEN_OUTER * 8, 256), S.bf16),
    kr_ws: S.Tensor((VARLEN_OUTER * 8, 256), S.bf16),
    gt_ws: S.Tensor((VARLEN_OUTER, DIM), S.bf16),
    inv_ws: S.Tensor((VARLEN_OUTER * CHUNK * CHUNK,), S.bf16),
    mqk_ws: S.Tensor((VARLEN_OUTER * CHUNK * CHUNK,), S.bf16),
    beta_ws: S.Tensor((VARLEN_OUTER * CHUNK,), S.bf16),
):
    tid = S.thread_id(0)
    warp = tid >> 5
    lane = tid & 31
    tile = S.block_id(0)
    head = S.block_id(1)

    sequence_start = 0
    sequence_tile_base = 0
    sequence_length = 1300
    if tile >= 82:
        sequence_start = 1300
        sequence_tile_base = 82
        sequence_length = 547
    if tile >= 117:
        sequence_start = 1847
        sequence_tile_base = 117
        sequence_length = 2048
    if tile >= 245:
        sequence_start = 3895
        sequence_tile_base = 245
        sequence_length = 963
    if tile >= 306:
        sequence_start = 4858
        sequence_tile_base = 306
        sequence_length = 271
    if tile >= 323:
        sequence_start = 5129
        sequence_tile_base = 323
        sequence_length = 3063
    local_tile = tile - sequence_tile_base
    t0 = sequence_start + local_tile * CHUNK
    remaining = sequence_length - local_tile * CHUNK
    actual_len = S.select(remaining < CHUNK, remaining, CHUNK)
    ws_idx = head * VARLEN_TILES + tile

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
    vector_layout = S.make_layout((BATCH * SEQUENCE * HEADS * DIM // 8, 4), (4, 1))
    q_vectors = S.view(q, S.u32, vector_layout)
    k_vectors = S.view(k, S.u32, vector_layout)
    g_vectors = S.view(g, S.u32, vector_layout)
    qk_stage_vectors = S.view(
        qk_stage,
        S.u32,
        S.make_layout((2 * CHUNK * PREP_STRIDE // 8, 4), (4, 1)),
    )
    g_stage_vectors = S.view(
        g_stage,
        S.u32,
        S.make_layout((CHUNK * PREP_STRIDE // 8, 4), (4, 1)),
    )
    zero_vector = S.full((1, 4), 0, S.u32)

    kd_desc = S.nvvm.make_tma_descriptor(kd_ws, tma_wide_layout)
    qd_desc = S.nvvm.make_tma_descriptor(qd_ws, tma_wide_layout)
    kr_desc = S.nvvm.make_tma_descriptor(kr_ws, tma_wide_layout)
    gt_desc = S.nvvm.make_tma_descriptor(gt_ws, S.make_layout((1, DIM), (DIM, 1)))

    row = tid >> 4
    col_vec = tid & 15
    shared_vector = row * (PREP_STRIDE // 8) + col_vec
    if row < actual_len:
        global_vector = ((t0 + row) * HEADS + head) * (DIM // 8) + col_vec
        q_vector = q_vectors[global_vector]
        k_vector = k_vectors[global_vector]
        g_vector = g_vectors[global_vector]
        qk_stage_vectors[CHUNK * PREP_STRIDE // 8 + shared_vector] = q_vector
        qk_stage_vectors[shared_vector] = k_vector
        g_stage_vectors[shared_vector] = g_vector
    else:
        zero = zero_vector[0]
        qk_stage_vectors[CHUNK * PREP_STRIDE // 8 + shared_vector] = zero
        qk_stage_vectors[shared_vector] = zero
        g_stage_vectors[shared_vector] = zero
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
        gate0 = S.convert(0.0, S.f32)
        gate1 = S.convert(0.0, S.f32)
        if row0 < actual_len:
            gv0 = S.convert(g_stage[row0, gate_col], S.f32) + dt
            gate0 = S.convert(GATE_SCALE, S.f32) * (
                S.nvvm.fast_tanh(S.convert(0.5, S.f32) * a_exp * gv0) * S.convert(0.5, S.f32) + S.convert(0.5, S.f32)
            )
        if row1 < actual_len:
            gv1 = S.convert(g_stage[row1, gate_col], S.f32) + dt
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

    inter_row = tid & 15
    inter_col = (tid >> 4) * 8
    for element_in_vector in S.range(8, unroll=True):
        kd_tma_raw[
            (tid * 8 + element_in_vector) // 256,
            (tid * 8 + element_in_vector) & 255,
        ] = qk_stage[0, inter_row, inter_col + element_in_vector]
        qd_tma_raw[
            (tid * 8 + element_in_vector) // 256,
            (tid * 8 + element_in_vector) & 255,
        ] = qk_stage[1, inter_row, inter_col + element_in_vector]
    S.syncthreads()
    if tid == 0:
        S.nvvm.fence_proxy_async_shared_cta()
        S.nvvm.tma_store(kd_tma_raw, kd_desc, (0, ws_idx * 8))
        S.nvvm.tma_store(qd_tma_raw, qd_desc, (0, ws_idx * 8))
        S.nvvm.cp_async_bulk_commit_group()

    if tid >= 64 and tid < 80:
        beta_row = tid - 64
        sigmoid = S.convert(0.0, S.f32)
        if beta_row < actual_len:
            beta_value = S.convert(beta[0, t0 + beta_row, head], S.f32)
            sigmoid = S.nvvm.fast_tanh(S.convert(0.5, S.f32) * beta_value) * S.convert(0.5, S.f32) + S.convert(
                0.5, S.f32
            )
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
                S.select(result_row < result_col, S.convert(0.0, S.f32), c0[0]),
                S.bf16,
            )
            mqk[result_row, result_col + 1] = S.convert(
                S.select(
                    result_row < result_col + 1,
                    S.convert(0.0, S.f32),
                    c0[1],
                ),
                S.bf16,
            )
            mqk[result_row + 8, result_col] = S.convert(
                S.select(
                    result_row + 8 < result_col,
                    S.convert(0.0, S.f32),
                    c0[2],
                ),
                S.bf16,
            )
            mqk[result_row + 8, result_col + 1] = S.convert(
                S.select(
                    result_row + 8 < result_col + 1,
                    S.convert(0.0, S.f32),
                    c0[3],
                ),
                S.bf16,
            )
            mqk[result_row, result_col + 8] = S.convert(
                S.select(
                    result_row < result_col + 8,
                    S.convert(0.0, S.f32),
                    c1[0],
                ),
                S.bf16,
            )
            mqk[result_row, result_col + 9] = S.convert(
                S.select(
                    result_row < result_col + 9,
                    S.convert(0.0, S.f32),
                    c1[1],
                ),
                S.bf16,
            )
            mqk[result_row + 8, result_col + 8] = S.convert(
                S.select(
                    result_row + 8 < result_col + 8,
                    S.convert(0.0, S.f32),
                    c1[2],
                ),
                S.bf16,
            )
            mqk[result_row + 8, result_col + 9] = S.convert(
                S.select(
                    result_row + 8 < result_col + 9,
                    S.convert(0.0, S.f32),
                    c1[3],
                ),
                S.bf16,
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


@avelang.jit
def _kda_recurrence_varlen_mix6_h64_dualcol(
    v: S.Tensor((BATCH, SEQUENCE, HEADS, DIM), S.bf16),
    out: S.Tensor((BATCH, SEQUENCE, HEADS, DIM), S.bf16),
    cu_seqlens: S.Tensor((VARLEN_SEQUENCES + 1,), S.i64),
    kd_ws: S.Tensor((VARLEN_OUTER * 8, 256), S.bf16),
    qd_ws: S.Tensor((VARLEN_OUTER * 8, 256), S.bf16),
    kr_ws: S.Tensor((VARLEN_OUTER * 8, 256), S.bf16),
    gt_ws: S.Tensor((VARLEN_OUTER, DIM), S.bf16),
    inv_ws: S.Tensor((VARLEN_OUTER * CHUNK * CHUNK,), S.bf16),
    mqk_ws: S.Tensor((VARLEN_OUTER * CHUNK * CHUNK,), S.bf16),
    beta_ws: S.Tensor((VARLEN_OUTER * CHUNK,), S.bf16),
):
    threads = 160
    tid = S.thread_id(0)
    warp = tid >> 5
    lane = tid & 31
    head = S.block_id(0)
    scheduled_sequence = S.block_id(1)
    tma_barrier = S.nvvm.mbarrier_create(2)

    # Process longer sequences first so the final waves contain the least work.
    sequence_index = 5
    sequence_tile_base = 323
    sequence_tiles = 192
    if scheduled_sequence == 1:
        sequence_index = 2
        sequence_tile_base = 117
        sequence_tiles = 128
    elif scheduled_sequence == 2:
        sequence_index = 0
        sequence_tile_base = 0
        sequence_tiles = 82
    elif scheduled_sequence == 3:
        sequence_index = 3
        sequence_tile_base = 245
        sequence_tiles = 61
    elif scheduled_sequence == 4:
        sequence_index = 1
        sequence_tile_base = 82
        sequence_tiles = 35
    elif scheduled_sequence == 5:
        sequence_index = 4
        sequence_tile_base = 306
        sequence_tiles = 17
    sequence_start = cu_seqlens[sequence_index]
    sequence_length = cu_seqlens[sequence_index + 1] - sequence_start

    u_tile = S.make_shared((8, CHUNK, U_STRIDE), S.bf16, 16)
    gt_stage = S.make_shared((2, DIM), S.bf16, 128)
    beta_stage = S.make_shared((2, CHUNK), S.bf16, 16)
    mqk_stage = S.make_shared((2, CHUNK, CHUNK), S.bf16, 16)
    inv_stage = S.make_shared((2, CHUNK, CHUNK), S.bf16, 16)
    v_stage = S.make_shared((2, CHUNK, V_STRIDE), S.bf16, 16)
    kr_stage = S.make_shared((2, 8, 256), S.bf16, 128)
    qd_stage = S.make_shared((2, CHUNK, DIM), S.bf16, 16)
    kd_stage = S.make_shared((2, CHUNK, DIM), S.bf16, 16)
    state = S.make_shared((8 * 8, CHUNK, STATE_STRIDE), S.bf16, 16)
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
        S.make_layout((8 * 8 * STATE_TILE_ELEMS // 2,), (1,)),
    )
    u_u32 = S.view(
        u_tile,
        S.u32,
        S.make_layout((8 * U_BLOCK_ELEMS // 2,), (1,)),
    )
    v_u32 = S.view(
        v_stage,
        S.u32,
        S.make_layout((2 * CHUNK * V_STRIDE // 2,), (1,)),
    )
    out_u32 = S.view(
        out,
        S.u32,
        S.make_layout((BATCH * SEQUENCE * HEADS * DIM // 2,), (1,)),
    )
    kr_desc = S.nvvm.make_tma_descriptor(kr_ws, S.make_layout((8, 256), (256, 1)))
    gt_desc = S.nvvm.make_tma_descriptor(gt_ws, S.make_layout((1, DIM), (DIM, 1)))
    for barrier_stage in S.range(2, unroll=True):
        S.nvvm.mbarrier_init(tma_barrier, barrier_stage, count=1, predicate=tid == 0)
    S.syncthreads()

    # Zero the complete padded state once. The padding columns make every
    # ldmatrix operand legal without per-tile clearing.
    for word in S.range(tid, 8 * 8 * STATE_TILE_ELEMS // 2, threads):
        state_u32[word] = 0
    S.syncthreads()

    ws_idx = head * VARLEN_TILES + sequence_tile_base
    workspace_byte = ws_idx * CHUNK * DIM * 2
    initial_actual_len = S.select(sequence_length < CHUNK, sequence_length, CHUNK)
    for copy in S.range(tid, 256, threads):
        S.nvvm.cp_async_cg_shared_global(kd_stage, kd_ws, copy * 16, workspace_byte + copy * 16, 16)
        S.nvvm.cp_async_cg_shared_global(qd_stage, qd_ws, copy * 16, workspace_byte + copy * 16, 16)
        v_row = copy >> 4
        v_col_vec = copy & 15
        v_destination = v_row * V_STRIDE + v_col_vec * 8
        if v_row < initial_actual_len:
            S.nvvm.cp_async_cg_shared_global(
                v_stage,
                v,
                v_destination * 2,
                (((sequence_start + v_row) * HEADS + head) * DIM + v_col_vec * 8) * 2,
                16,
            )
        else:
            for pair_in_vector in S.range(4, unroll=True):
                v_u32[(v_destination >> 1) + pair_in_vector] = 0
        if copy < 32:
            S.nvvm.cp_async_cg_shared_global(inv_stage, inv_ws, copy * 16, (ws_idx * 256 * 2) + copy * 16, 16)
            S.nvvm.cp_async_cg_shared_global(mqk_stage, mqk_ws, copy * 16, (ws_idx * 256 * 2) + copy * 16, 16)
    if tid == 0:
        S.nvvm.cp_async_cg_shared_global(beta_stage, beta_ws, 0, ws_idx * CHUNK * 2, 16)
    elif tid == 1:
        S.nvvm.cp_async_cg_shared_global(beta_stage, beta_ws, 16, ws_idx * CHUNK * 2 + 16, 16)
    S.nvvm.mbarrier_arrive_expect_tx(tma_barrier, (CHUNK * DIM + DIM) * 2, 0, tid == 0)
    kr_initial_raw = S.subview(kr_stage, (0, 0, 0), (1, 8, 256), (1, 1, 1))
    kr_initial = S.view(kr_initial_raw, S.bf16, S.make_layout((8, 256), (256, 1)))
    S.nvvm.tma_load(
        kr_initial,
        kr_desc,
        (0, ws_idx * 8),
        tma_barrier,
        mbar_id=0,
        predicate=tid == 0,
        expect_tx=False,
    )
    gt_initial_raw = S.subview(gt_stage, (0, 0), (1, DIM), (1, 1))
    gt_initial = S.view(gt_initial_raw, S.bf16, S.make_layout((1, DIM), (DIM, 1)))
    S.nvvm.tma_load(
        gt_initial,
        gt_desc,
        (0, ws_idx),
        tma_barrier,
        mbar_id=0,
        predicate=tid == 0,
        expect_tx=False,
    )
    S.nvvm.cp_async_commit_group()
    S.nvvm.cp_async_wait_group(0)
    S.syncthreads()

    for tile in S.range(sequence_tiles):
        stage = tile & 1
        t0 = sequence_start + tile * CHUNK
        remaining = sequence_length - tile * CHUNK
        actual_len = S.select(remaining < CHUNK, remaining, CHUNK)

        if tile + 1 < sequence_tiles:
            next_tile = tile + 1
            next_stage = next_tile & 1
            next_ws = head * VARLEN_TILES + sequence_tile_base + next_tile
            next_tile_offset = next_stage * CHUNK * DIM
            next_workspace_byte = next_ws * CHUNK * DIM * 2
            next_remaining = sequence_length - next_tile * CHUNK
            next_actual_len = S.select(next_remaining < CHUNK, next_remaining, CHUNK)
            for copy in S.range(tid, 256, threads):
                S.nvvm.cp_async_cg_shared_global(
                    kd_stage,
                    kd_ws,
                    next_tile_offset * 2 + copy * 16,
                    next_workspace_byte + copy * 16,
                    16,
                )
                S.nvvm.cp_async_cg_shared_global(
                    qd_stage,
                    qd_ws,
                    next_tile_offset * 2 + copy * 16,
                    next_workspace_byte + copy * 16,
                    16,
                )
                v_row = copy >> 4
                v_col_vec = copy & 15
                v_destination = next_stage * CHUNK * V_STRIDE + v_row * V_STRIDE + v_col_vec * 8
                if v_row < next_actual_len:
                    S.nvvm.cp_async_cg_shared_global(
                        v_stage,
                        v,
                        v_destination * 2,
                        (((sequence_start + next_tile * CHUNK + v_row) * HEADS + head) * DIM + v_col_vec * 8) * 2,
                        16,
                    )
                else:
                    for pair_in_vector in S.range(4, unroll=True):
                        v_u32[(v_destination >> 1) + pair_in_vector] = 0
                if copy < 32:
                    S.nvvm.cp_async_cg_shared_global(
                        inv_stage,
                        inv_ws,
                        (next_stage * 256) * 2 + copy * 16,
                        (next_ws * 256) * 2 + copy * 16,
                        16,
                    )
                    S.nvvm.cp_async_cg_shared_global(
                        mqk_stage,
                        mqk_ws,
                        (next_stage * 256) * 2 + copy * 16,
                        (next_ws * 256) * 2 + copy * 16,
                        16,
                    )
            if tid == 0:
                S.nvvm.cp_async_cg_shared_global(
                    beta_stage,
                    beta_ws,
                    (next_stage * CHUNK) * 2,
                    next_ws * CHUNK * 2,
                    16,
                )
            elif tid == 1:
                S.nvvm.cp_async_ca_shared_global(
                    beta_stage,
                    beta_ws,
                    (next_stage * CHUNK) * 2 + 16,
                    next_ws * CHUNK * 2 + 16,
                    16,
                )
            S.nvvm.mbarrier_arrive_expect_tx(
                tma_barrier,
                (CHUNK * DIM + DIM) * 2,
                next_stage,
                tid == 0,
            )
            kr_next_raw = S.subview(kr_stage, (next_stage, 0, 0), (1, 8, 256), (1, 1, 1))
            kr_next = S.view(kr_next_raw, S.bf16, S.make_layout((8, 256), (256, 1)))
            S.nvvm.tma_load(
                kr_next,
                kr_desc,
                (0, next_ws * 8),
                tma_barrier,
                mbar_id=next_stage,
                predicate=tid == 0,
                expect_tx=False,
            )
            gt_next_raw = S.subview(gt_stage, (next_stage, 0), (1, DIM), (1, 1))
            gt_next = S.view(gt_next_raw, S.bf16, S.make_layout((1, DIM), (DIM, 1)))
            S.nvvm.tma_load(
                gt_next,
                gt_desc,
                (0, next_ws),
                tma_barrier,
                mbar_id=next_stage,
                predicate=tid == 0,
                expect_tx=False,
            )
            S.nvvm.cp_async_commit_group()

        if warp < 4:
            k_c0 = S.full((2, 4), 0.0, S.f32)
            k_c1 = S.full((2, 4), 0.0, S.f32)
            q_c0 = S.full((2, 4), 0.0, S.f32)
            q_c1 = S.full((2, 4), 0.0, S.f32)
            matrix_row = (((lane >> 3) & 1) << 3) + (lane & 7)
            matrix_col = (lane >> 4) << 3
            for k_block in S.range(8, unroll=True):
                inter_address = (lane & 7) * 8 + ((lane >> 3) & 1) * 64 + (k_block * 2 + (lane >> 4)) * 128
                a_k = S.nvvm.ldmatrix_m8n8_x4_b16(kd_smem + (stage * CHUNK * DIM + inter_address) * 2)
                a_q = S.nvvm.ldmatrix_m8n8_x4_b16(qd_smem + (stage * CHUNK * DIM + inter_address) * 2)
                for column_iteration in S.range(2, unroll=True):
                    col_block = warp * 2 + column_iteration
                    state_block = col_block + k_block * 8
                    state_address = (
                        state_smem + (state_block * STATE_TILE_ELEMS + matrix_row * STATE_STRIDE + matrix_col) * 2
                    )
                    b_state = S.nvvm.ldmatrix_m8n8_x4_b16(state_address)
                    b_lo = S.full((2,), 0, S.i32)
                    b_hi = S.full((2,), 0, S.i32)
                    b_lo[0] = b_state[0]
                    b_lo[1] = b_state[2]
                    b_hi[0] = b_state[1]
                    b_hi[1] = b_state[3]
                    k0 = k_c0[column_iteration]
                    k1 = k_c1[column_iteration]
                    q0 = q_c0[column_iteration]
                    q1 = q_c1[column_iteration]
                    k0 = S.nvvm.mma_16x8x16_bf16_f32(a_k, b_lo, k0)
                    k1 = S.nvvm.mma_16x8x16_bf16_f32(a_k, b_hi, k1)
                    q0 = S.nvvm.mma_16x8x16_bf16_f32(a_q, b_lo, q0)
                    q1 = S.nvvm.mma_16x8x16_bf16_f32(a_q, b_hi, q1)
                    for element in S.range(4, unroll=True):
                        k_c0[column_iteration, element] = k0[element]
                        k_c1[column_iteration, element] = k1[element]
                        q_c0[column_iteration, element] = q0[element]
                        q_c1[column_iteration, element] = q1[element]

            result_row = lane >> 2
            result_col = (lane & 3) << 1
            u_top = result_row * U_STRIDE + result_col
            u_bottom = (result_row + 8) * U_STRIDE + result_col
            beta_top = S.convert(beta_stage[stage, result_row], S.f32)
            beta_bottom = S.convert(beta_stage[stage, result_row + 8], S.f32)
            beta_top_pair = S.nvvm.floatx2_to_bf16x2(beta_top, beta_top)
            beta_bottom_pair = S.nvvm.floatx2_to_bf16x2(beta_bottom, beta_bottom)
            q_pairs = S.full((2, 4), 0, S.i32)
            for column_iteration in S.range(2, unroll=True):
                col_block = warp * 2 + column_iteration
                col0 = col_block * 16
                v_address = v_smem + (stage * CHUNK * V_STRIDE + matrix_row * V_STRIDE + col0 + matrix_col) * 2
                v_frag = S.nvvm.ldmatrix_m8n8_x4_b16(v_address)
                u_base = col_block * U_BLOCK_ELEMS
                k_pair0 = S.nvvm.floatx2_to_bf16x2(k_c0[column_iteration, 0], k_c0[column_iteration, 1])
                k_pair1 = S.nvvm.floatx2_to_bf16x2(k_c0[column_iteration, 2], k_c0[column_iteration, 3])
                k_pair2 = S.nvvm.floatx2_to_bf16x2(k_c1[column_iteration, 0], k_c1[column_iteration, 1])
                k_pair3 = S.nvvm.floatx2_to_bf16x2(k_c1[column_iteration, 2], k_c1[column_iteration, 3])
                u_u32[(u_base + u_top) >> 1] = S.nvvm.bf16x2_mul(S.nvvm.bf16x2_sub(v_frag[0], k_pair0), beta_top_pair)
                u_u32[(u_base + u_bottom) >> 1] = S.nvvm.bf16x2_mul(
                    S.nvvm.bf16x2_sub(v_frag[1], k_pair1), beta_bottom_pair
                )
                u_u32[(u_base + u_top + 8) >> 1] = S.nvvm.bf16x2_mul(
                    S.nvvm.bf16x2_sub(v_frag[2], k_pair2), beta_top_pair
                )
                u_u32[(u_base + u_bottom + 8) >> 1] = S.nvvm.bf16x2_mul(
                    S.nvvm.bf16x2_sub(v_frag[3], k_pair3), beta_bottom_pair
                )
                q_pairs[column_iteration, 0] = S.nvvm.floatx2_to_bf16x2(
                    q_c0[column_iteration, 0], q_c0[column_iteration, 1]
                )
                q_pairs[column_iteration, 1] = S.nvvm.floatx2_to_bf16x2(
                    q_c0[column_iteration, 2], q_c0[column_iteration, 3]
                )
                q_pairs[column_iteration, 2] = S.nvvm.floatx2_to_bf16x2(
                    q_c1[column_iteration, 0], q_c1[column_iteration, 1]
                )
                q_pairs[column_iteration, 3] = S.nvvm.floatx2_to_bf16x2(
                    q_c1[column_iteration, 2], q_c1[column_iteration, 3]
                )

            inv_address = (matrix_row & 7) * 8 + (matrix_row >> 3) * 64 + (matrix_col & 7) + (matrix_col >> 3) * 128
            inv_a = S.nvvm.ldmatrix_m8n8_x4_b16(inv_smem + (stage * 256 + inv_address) * 2)
            for column_iteration in S.range(2, unroll=True):
                col_block = warp * 2 + column_iteration
                u_base = col_block * U_BLOCK_ELEMS
                u_address = u_smem + (u_base + matrix_row * U_STRIDE + matrix_col) * 2
                u_b = S.nvvm.ldmatrix_m8n8_x4_b16_trans(u_address)
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

            mqk_a = S.nvvm.ldmatrix_m8n8_x4_b16(mqk_smem + (stage * 256 + inv_address) * 2)
            for column_iteration in S.range(2, unroll=True):
                col_block = warp * 2 + column_iteration
                col0 = col_block * 16
                u_base = col_block * U_BLOCK_ELEMS
                u_address = u_smem + (u_base + matrix_row * U_STRIDE + matrix_col) * 2
                u_b = S.nvvm.ldmatrix_m8n8_x4_b16_trans(u_address)
                u_b_lo = S.full((2,), 0, S.i32)
                u_b_hi = S.full((2,), 0, S.i32)
                u_b_lo[0] = u_b[0]
                u_b_lo[1] = u_b[1]
                u_b_hi[0] = u_b[2]
                u_b_hi[1] = u_b[3]
                out_c0 = S.full((4,), 0.0, S.f32)
                out_c1 = S.full((4,), 0.0, S.f32)
                out_c0 = S.nvvm.mma_16x8x16_bf16_f32(mqk_a, u_b_lo, out_c0)
                out_c1 = S.nvvm.mma_16x8x16_bf16_f32(mqk_a, u_b_hi, out_c1)
                out_pair0 = S.nvvm.floatx2_to_bf16x2(out_c0[0], out_c0[1])
                out_pair1 = S.nvvm.floatx2_to_bf16x2(out_c0[2], out_c0[3])
                out_pair2 = S.nvvm.floatx2_to_bf16x2(out_c1[0], out_c1[1])
                out_pair3 = S.nvvm.floatx2_to_bf16x2(out_c1[2], out_c1[3])
                output_base = ((t0 + result_row) * HEADS + head) * DIM
                output_bottom = ((t0 + result_row + 8) * HEADS + head) * DIM
                if result_row < actual_len:
                    out_u32[(output_base + col0 + result_col) >> 1] = S.nvvm.bf16x2_add(
                        q_pairs[column_iteration, 0], out_pair0
                    )
                    out_u32[(output_base + col0 + result_col + 8) >> 1] = S.nvvm.bf16x2_add(
                        q_pairs[column_iteration, 2], out_pair2
                    )
                if result_row + 8 < actual_len:
                    out_u32[(output_bottom + col0 + result_col) >> 1] = S.nvvm.bf16x2_add(
                        q_pairs[column_iteration, 1], out_pair1
                    )
                    out_u32[(output_bottom + col0 + result_col + 8) >> 1] = S.nvvm.bf16x2_add(
                        q_pairs[column_iteration, 3], out_pair3
                    )

        tma_phase = (tile >> 1) & 1
        S.nvvm.mbarrier_try_wait_parity(tma_barrier, tma_phase, 10000000, stage)
        S.syncthreads()

        if warp < 4:
            matrix_row = (((lane >> 3) & 1) << 3) + (lane & 7)
            matrix_col = (lane >> 4) << 3
            result_col = (lane & 3) << 1
            update_b_fragments = S.full((2, 4), 0, S.i32)
            update_gt_pairs = S.full((2, 2), 0, S.i32)
            for column_iteration in S.range(2, unroll=True):
                col_block = warp * 2 + column_iteration
                kr_address = col_block * 256 + matrix_row * 16 + matrix_col
                update_b = S.nvvm.ldmatrix_m8n8_x4_b16_trans(kr_smem + (stage * CHUNK * DIM + kr_address) * 2)
                for element in S.range(4, unroll=True):
                    update_b_fragments[column_iteration, element] = update_b[element]
                gt_col = col_block * 16 + result_col
                update_gt_pairs[column_iteration, 0] = S.nvvm.floatx2_to_bf16x2(
                    S.convert(gt_stage[stage, gt_col], S.f32),
                    S.convert(gt_stage[stage, gt_col + 1], S.f32),
                )
                update_gt_pairs[column_iteration, 1] = S.nvvm.floatx2_to_bf16x2(
                    S.convert(gt_stage[stage, gt_col + 8], S.f32),
                    S.convert(gt_stage[stage, gt_col + 9], S.f32),
                )
            for local_row_block in S.range(8, unroll=True):
                update_a = S.nvvm.ldmatrix_m8n8_x4_b16_trans(
                    u_smem + (local_row_block * U_BLOCK_ELEMS + matrix_row * U_STRIDE + matrix_col) * 2
                )
                for column_iteration in S.range(2, unroll=True):
                    col_block = warp * 2 + column_iteration
                    b_lo = S.full((2,), 0, S.i32)
                    b_hi = S.full((2,), 0, S.i32)
                    b_lo[0] = update_b_fragments[column_iteration, 0]
                    b_lo[1] = update_b_fragments[column_iteration, 1]
                    b_hi[0] = update_b_fragments[column_iteration, 2]
                    b_hi[1] = update_b_fragments[column_iteration, 3]
                    state_block = local_row_block * 8 + col_block
                    state_frag = S.nvvm.ldmatrix_m8n8_x4_b16(
                        state_smem + (state_block * STATE_TILE_ELEMS + matrix_row * STATE_STRIDE + matrix_col) * 2
                    )
                    update_c0 = S.full((4,), 0.0, S.f32)
                    update_c1 = S.full((4,), 0.0, S.f32)
                    update_c0 = S.nvvm.mma_16x8x16_bf16_f32(update_a, b_lo, update_c0)
                    update_c1 = S.nvvm.mma_16x8x16_bf16_f32(update_a, b_hi, update_c1)
                    state_row = lane >> 2
                    state_col = (lane & 3) << 1
                    state_base = state_block * STATE_TILE_ELEMS
                    state_top = state_base + state_row * STATE_STRIDE + state_col
                    state_bottom = state_base + (state_row + 8) * STATE_STRIDE + state_col
                    update_pair0 = S.nvvm.floatx2_to_bf16x2(update_c0[0], update_c0[1])
                    update_pair1 = S.nvvm.floatx2_to_bf16x2(update_c0[2], update_c0[3])
                    update_pair2 = S.nvvm.floatx2_to_bf16x2(update_c1[0], update_c1[1])
                    update_pair3 = S.nvvm.floatx2_to_bf16x2(update_c1[2], update_c1[3])
                    state_u32[state_top >> 1] = S.nvvm.bf16x2_fma(
                        state_frag[0], update_gt_pairs[column_iteration, 0], update_pair0
                    )
                    state_u32[state_bottom >> 1] = S.nvvm.bf16x2_fma(
                        state_frag[1], update_gt_pairs[column_iteration, 0], update_pair1
                    )
                    state_u32[(state_top + 8) >> 1] = S.nvvm.bf16x2_fma(
                        state_frag[2], update_gt_pairs[column_iteration, 1], update_pair2
                    )
                    state_u32[(state_bottom + 8) >> 1] = S.nvvm.bf16x2_fma(
                        state_frag[3], update_gt_pairs[column_iteration, 1], update_pair3
                    )

        if tile + 1 < sequence_tiles:
            S.nvvm.cp_async_wait_group(0)
            S.syncthreads()


def get_workspace_size(varlen: bool = False) -> int:
    """Return the selected specialization's workspace size in bytes."""

    outer = VARLEN_OUTER if varlen else OUTER
    elements = 3 * outer * CHUNK * DIM + outer * DIM + 2 * outer * CHUNK * CHUNK + outer * CHUNK
    return elements * 2


def _workspace_views(workspace: torch.Tensor, outer: int = OUTER):
    offset = 0

    def take(elements, shape):
        nonlocal offset
        result = workspace.narrow(0, offset, elements).view(shape)
        offset += elements
        return result

    tile_elements = outer * CHUNK * DIM
    kd = take(tile_elements, (outer * 8, 256))
    qd = take(tile_elements, (outer * 8, 256))
    kr = take(tile_elements, (outer * 8, 256))
    gt = take(outer * DIM, (outer, DIM))
    inv = take(outer * CHUNK * CHUNK, (outer * CHUNK * CHUNK,))
    mqk = take(outer * CHUNK * CHUNK, (outer * CHUNK * CHUNK,))
    beta = take(outer * CHUNK, (outer * CHUNK,))
    return kd, qd, kr, gt, inv, mqk, beta


def _validate_varlen_metadata(cu_seqlens: torch.Tensor):
    if (
        cu_seqlens.dtype != torch.int64
        or cu_seqlens.device.type != "cuda"
        or not cu_seqlens.is_contiguous()
        or tuple(cu_seqlens.shape) != (VARLEN_SEQUENCES + 1,)
    ):
        raise ValueError("cu_seqlens must be contiguous CUDA int64 with shape (7,)")
    key = (
        cu_seqlens.device.index,
        cu_seqlens.data_ptr(),
        cu_seqlens._version,
    )
    if key not in _verified_varlen_metadata:
        values = tuple(int(value) for value in cu_seqlens.tolist())
        if values != VARLEN_CU_SEQLENS:
            raise ValueError(
                f"the selected varlen specialization requires cumulative lengths {VARLEN_CU_SEQLENS}, got {values}"
            )
        _verified_varlen_metadata.add(key)


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
    """Run the fixed or selected six-sequence packed KDA specialization."""

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
    if initial_state is not None or final_state is not None:
        raise ValueError("the selected specializations do not accept state")
    if not math.isclose(float(scale), SCALE, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"scale must be 1/sqrt(128), got {scale}")
    if float(lower_bound) != -5.0:
        raise ValueError("lower_bound must be -5.0")

    if cu_seqlens is not None:
        _validate_varlen_metadata(cu_seqlens)
        workspace_elements = get_workspace_size(varlen=True) // 2
        workspace = torch.empty(workspace_elements, dtype=torch.bfloat16, device=q.device)
        kd, qd, kr, gt, inv, mqk, beta_act = _workspace_views(workspace, VARLEN_OUTER)
        _kda_prepare_varlen_mix6_h64[lambda: ((VARLEN_TILES, HEADS, 1), (THREADS, 1, 1))](
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
        _kda_recurrence_varlen_mix6_h64_dualcol[lambda: ((HEADS, VARLEN_SEQUENCES, 1), (160, 1, 1))](
            v,
            out,
            cu_seqlens,
            kd,
            qd,
            kr,
            gt,
            inv,
            mqk,
            beta_act,
            num_warps=5,
            min_ctas=2,
            prefer_l1=False,
        )
        return out

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
