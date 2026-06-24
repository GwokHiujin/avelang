#!/usr/bin/env python3
import argparse
import importlib.util
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_KERNEL_PATH = (
    REPO_ROOT / "test/examples/gemm/nvidia/test_gemm_wgmma_bf16_square.py"
)


def _load_kernel_module():
    spec = importlib.util.spec_from_file_location(
        "test_gemm_wgmma_bf16_square", TEST_KERNEL_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_wgmma_device():
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


def _repeat_for_size(size):
    if size <= 1024:
        return 200
    if size <= 2048:
        return 100
    if size <= 4096:
        return 50
    if size <= 8192:
        return 20
    return 5


def run_square_benchmark(size, warmup, repeat, iters, validate):
    kernel_mod = _load_kernel_module()
    device_idx = _get_wgmma_device()
    if device_idx is None:
        raise RuntimeError("No Hopper-or-newer CUDA device found for WGMMA.")

    if size % kernel_mod.TILE_M != 0 or size % kernel_mod.TILE_N != 0:
        raise ValueError(
            f"size must be a multiple of {kernel_mod.TILE_M}x{kernel_mod.TILE_N}, got {size}."
        )

    torch.cuda.set_device(device_idx)
    device = torch.device(f"cuda:{device_idx}")

    a = torch.ones((size, size), dtype=torch.bfloat16, device=device)
    b = torch.ones((size, size), dtype=torch.bfloat16, device=device)
    c = torch.zeros((size, size), dtype=torch.bfloat16, device=device)

    grid = (size // kernel_mod.TILE_N, size // kernel_mod.TILE_M, 1)
    block = (kernel_mod.THREADS_PER_CTA, 1, 1)

    for _ in range(warmup):
        kernel_mod.gemm_wgmma_bf16_square_kernel[lambda: (grid, block)](
            a, b, c, size
        )
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=capture_stream):
        kernel_mod.gemm_wgmma_bf16_square_kernel[lambda: (grid, block)](
            a, b, c, size
        )
    torch.cuda.synchronize(device)

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(repeat):
        for _ in range(iters):
            graph.replay()
    end_evt.record()
    torch.cuda.synchronize(device)

    elapsed_ms = start_evt.elapsed_time(end_evt)
    elapsed_s = (elapsed_ms * 1.0e-3) / (repeat * iters)
    flops = 2.0 * size * size * size
    tflops = flops / elapsed_s / 1.0e12

    bytes_moved = (size * size * 3) * 2
    bandwidth_gbs = bytes_moved / elapsed_s / 1.0e9

    print(
        f"M={size} N={size} K={size} time_ms={elapsed_s * 1.0e3:.4f} "
        f"tflops={tflops:.3f} bandwidth_gbs={bandwidth_gbs:.3f} "
        f"device={device_idx}"
    )

    if validate:
        expected = torch.full_like(c, float(size))
        actual = c
        max_abs = torch.max(torch.abs(actual.float() - expected.float())).item()
        if not torch.equal(actual, expected):
            raise AssertionError(f"Validation failed (max abs diff {max_abs}).")
        print(f"validation=max_abs_diff:{max_abs:.6f}")

    return elapsed_s, tflops, bandwidth_gbs, device_idx


def main():
    parser = argparse.ArgumentParser(
        description="NVIDIA Hopper WGMMA BF16 square GEMM benchmark"
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192, 16384],
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--validate", action="store_true", default=False)
    args = parser.parse_args()

    for size in args.sizes:
        repeat = args.repeat if args.repeat is not None else _repeat_for_size(size)
        run_square_benchmark(size, args.warmup, repeat, args.iters, args.validate)


if __name__ == "__main__":
    main()
