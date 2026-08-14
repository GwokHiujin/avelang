#!/usr/bin/env python3
"""Benchmark Avelang's FP8 MoE GEMM pipeline with the CUDA test workload."""

import argparse
import gc

import torch
from avelang_kernels.nvidia_moe import moe_gemm_fp8, silu_mul_quant_fp8

TOKENS = (1024, 2048, 4096, 8192, 16384)
DIM = 7168
INTER_DIM = 2048
NUM_EXPERTS = 32
TOPK = 4
SEED = 20260319
WARMUP = 10
REPEAT = 100
GRAPH_ITERS = 1
MEASURE_TRIALS = 2
BLOCK_SIZE = 128


def _select_hopper(device=None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if torch.version.hip is not None:
        raise RuntimeError("NVIDIA CUDA is not available; found a ROCm build.")
    if device is None:
        device = next(
            (
                torch.device(f"cuda:{index}")
                for index in range(torch.cuda.device_count())
                if torch.cuda.get_device_capability(index)[0] == 9
            ),
            None,
        )
        if device is None:
            raise RuntimeError("No compute-capability 9.x CUDA device is available.")
    else:
        device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"--device must select CUDA, got {device}.")
    torch.cuda.set_device(device)
    major, minor = torch.cuda.get_device_capability(device)
    if major != 9:
        raise RuntimeError(
            f"NVIDIA MoE benchmark requires compute capability 9.x, got "
            f"{major}.{minor}."
        )
    return device


def _fill_fp8_normal_by_expert(target, generator):
    """Match the CUDA benchmark while bounding FP32 staging memory."""

    for expert_weights in target:
        staging = torch.randn(
            expert_weights.shape,
            dtype=torch.float32,
            device=target.device,
            generator=generator,
        ).mul_(8.0)
        expert_weights.copy_(staging)
        del staging


def build_weights(device):
    generator = torch.Generator(device=device)
    generator.manual_seed(SEED + 17)
    w13 = torch.empty(
        (NUM_EXPERTS, 2 * INTER_DIM, DIM),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    _fill_fp8_normal_by_expert(w13, generator)
    w2 = torch.empty(
        (NUM_EXPERTS, DIM, INTER_DIM),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    _fill_fp8_normal_by_expert(w2, generator)
    w13_scale = (
        torch.randn(
            (NUM_EXPERTS, (2 * INTER_DIM // 128) * (DIM // 128)),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        .mul_(2e-3)
        .add_(1e-2)
        .clamp_min_(1e-8)
        .view(NUM_EXPERTS, 2 * INTER_DIM // 128, DIM // 128)
        .contiguous()
    )
    w2_scale = (
        torch.randn(
            (NUM_EXPERTS, (DIM // 128) * (INTER_DIM // 128)),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        .mul_(2e-3)
        .add_(1e-2)
        .clamp_min_(1e-8)
        .view(NUM_EXPERTS, DIM // 128, INTER_DIM // 128)
        .contiguous()
    )
    return w13, w2, w13_scale, w2_scale


def build_routed_input(tokens, device, generator):
    """Generate and expert-permute the CUDA benchmark's shared input."""

    input_q = (
        torch.randn(
            (tokens, DIM),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        .add_(0.1)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    input_scale = (
        torch.randn(
            (tokens, DIM // 128),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        .mul_(2e-2)
        .add_(1e-1)
        .clamp_min_(1e-8)
        .contiguous()
    )
    scores = torch.randn(
        (tokens, NUM_EXPERTS),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    topk_ids = torch.topk(
        scores, k=TOPK, dim=-1, largest=True, sorted=True
    ).indices.to(torch.int32)

    expert_tokens = []
    padded_counts = []
    for expert in range(NUM_EXPERTS):
        token_ids = torch.nonzero(topk_ids == expert, as_tuple=True)[0]
        expert_tokens.append(token_ids)
        padded_counts.append(
            ((token_ids.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
        )
    routed_rows = sum(padded_counts)
    # The Avelang port launches two-CTA M clusters.  Add at most one empty
    # block so the physical workspace is cluster-aligned.
    max_rows = ((routed_rows + 255) // 256) * 256
    routed_input = torch.zeros(
        (max_rows, DIM), dtype=torch.float8_e4m3fn, device=device
    )
    routed_scale = torch.ones(
        (DIM // 128, max_rows), dtype=torch.float32, device=device
    )
    block_expert_ids = torch.zeros(
        (max_rows // 128,), dtype=torch.int32, device=device
    )
    offset = 0
    for expert, (token_ids, padded) in enumerate(
        zip(expert_tokens, padded_counts, strict=True)
    ):
        count = token_ids.numel()
        if count:
            routed_input[offset : offset + count].copy_(input_q[token_ids])
            routed_scale[:, offset : offset + count].copy_(
                input_scale[token_ids].T
            )
        block_expert_ids[offset // 128 : (offset + padded) // 128] = expert
        offset += padded

    del input_q, input_scale, scores, topk_ids
    return routed_input, routed_scale, block_expert_ids, routed_rows


def benchmark_with_cuda_graph(fn, warmup, repeat, graph_iters, measure_trials):
    """Use the CUDA benchmark's warmup and minimum-of-two timing protocol."""

    replay_warmup = max(1, min(5, warmup))
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        captured_outputs = []
        with torch.cuda.graph(graph):
            for _ in range(graph_iters):
                captured_outputs.append(fn())
        torch.cuda.synchronize()

        for _ in range(replay_warmup):
            graph.replay()
        torch.cuda.synchronize()

        elapsed_ms = []
        for _ in range(measure_trials):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeat):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            elapsed_ms.append(float(start.elapsed_time(end)))
    return min(elapsed_ms) / float(repeat * graph_iters)


def moe_tflops(mean_ms, tokens):
    flops = 6.0 * tokens * TOPK * DIM * INTER_DIM
    return flops / (mean_ms * 1e-3) / 1e12


def run_token_case(
    tokens,
    weights,
    input_generator,
    warmup,
    repeat,
    graph_iters,
    measure_trials,
):
    w13, w2, w13_scale, w2_scale = weights
    routed_input, routed_scale, block_expert_ids, routed_rows = (
        build_routed_input(tokens, w13.device, input_generator)
    )
    max_rows = routed_input.shape[0]
    gemm1_out = torch.empty(
        (max_rows, 2 * INTER_DIM), dtype=torch.bfloat16, device=w13.device
    )
    gemm2_input = torch.empty(
        (max_rows, INTER_DIM), dtype=torch.float8_e4m3fn, device=w13.device
    )
    gemm2_scale = torch.empty(
        (INTER_DIM // 128, max_rows), dtype=torch.float32, device=w13.device
    )
    gemm2_out = torch.empty(
        (max_rows, DIM), dtype=torch.bfloat16, device=w13.device
    )

    def call():
        moe_gemm_fp8(
            routed_input,
            w13,
            routed_scale,
            w13_scale,
            block_expert_ids,
            out=gemm1_out,
        )
        silu_mul_quant_fp8(gemm1_out, gemm2_input, gemm2_scale)
        return moe_gemm_fp8(
            gemm2_input,
            w2,
            gemm2_scale,
            w2_scale,
            block_expert_ids,
            out=gemm2_out,
        )

    mean_ms = benchmark_with_cuda_graph(
        call, warmup, repeat, graph_iters, measure_trials
    )
    print(
        f"tokens={tokens} dim={DIM} inter_dim={INTER_DIM} "
        f"experts={NUM_EXPERTS} topk={TOPK} routed_rows={routed_rows} "
        f"workspace_rows={max_rows} mean_ms={mean_ms:.4f} "
        f"tflops={moe_tflops(mean_ms, tokens):.3f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Avelang FP8 MoE benchmark using the CUDA test workload",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=list(TOKENS))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--repeat", type=int, default=REPEAT)
    parser.add_argument("--graph-iters", type=int, default=GRAPH_ITERS)
    parser.add_argument("--measure-trials", type=int, default=MEASURE_TRIALS)
    parser.add_argument(
        "--device",
        help="CUDA device (for example cuda:2); defaults to the first Hopper device",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dry_run:
        print(
            f"tokens={tuple(args.tokens)} dim={DIM} inter_dim={INTER_DIM} "
            f"experts={NUM_EXPERTS} topk={TOPK} seed={args.seed} "
            f"warmup={args.warmup} repeat={args.repeat} "
            f"graph_iters={args.graph_iters} "
            f"measure_trials={args.measure_trials} block_shape=(128, 128)"
        )
        return

    device = _select_hopper(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    weights = build_weights(device)
    input_generator = torch.Generator(device=device)
    input_generator.manual_seed(args.seed)
    for tokens in args.tokens:
        run_token_case(
            tokens,
            weights,
            input_generator,
            args.warmup,
            args.repeat,
            args.graph_iters,
            args.measure_trials,
        )
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
