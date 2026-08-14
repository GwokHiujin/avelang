#!/usr/bin/env python3
import argparse

import torch

from avelang_kernels.nvidia_gemm import SUPPORTED_SIZES, gemm_bf16


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
                if torch.cuda.get_device_capability(index)[0] >= 9
            ),
            None,
        )
        if device is None:
            raise RuntimeError("No Hopper-or-newer CUDA device is available.")
    else:
        device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"--device must select CUDA, got {device}.")
    torch.cuda.set_device(device)
    major, minor = torch.cuda.get_device_capability(device)
    if major < 9:
        raise RuntimeError(
            f"NVIDIA GEMM benchmark requires Hopper or newer, got "
            f"compute capability {major}.{minor}."
        )
    return device


def run_gemm_benchmark(size, warmup, repeat, iters, validate, device=None):
    device = _select_hopper(device)
    if size not in SUPPORTED_SIZES:
        raise ValueError(f"size must be one of {SUPPORTED_SIZES}, got {size}.")

    a = torch.randn((size, size), dtype=torch.bfloat16, device=device)
    b = torch.randn_like(a)
    output = torch.empty_like(a)

    # Warmup to JIT-compile and stabilize clocks.
    for _ in range(warmup):
        gemm_bf16(a, b, out=output)
    torch.cuda.synchronize()

    # Capture the kernel launch in a CUDAGraph for low-overhead replays.
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=capture_stream):
        gemm_bf16(a, b, out=output)
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(repeat):
        for _ in range(iters):
            graph.replay()
    end_evt.record()
    torch.cuda.synchronize()
    elapsed_ms = start_evt.elapsed_time(end_evt) / (repeat * iters)
    elapsed = elapsed_ms * 1.0e-3

    flops = 2.0 * size * size * size
    tflops = flops / elapsed / 1.0e12
    bytes_moved = 3 * size * size * 2
    bandwidth = bytes_moved / elapsed / 1.0e9

    print(
        f"M={size} N={size} K={size} time_ms={elapsed_ms:.4f} "
        f"tflops={tflops:.3f} bandwidth_gbs={bandwidth:.3f}"
    )

    if validate:
        expected = a @ b.T
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        max_abs = torch.max(torch.abs(output.float() - expected.float())).item()
        print(f"validation=max_abs_diff:{max_abs:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="NVIDIA Hopper BF16 square GEMM benchmark"
    )
    parser.add_argument("--size", type=int, choices=SUPPORTED_SIZES, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument(
        "--iters",
        type=int,
        default=1,
        help="graph replays per timing repeat (CUDA reference default: 1)",
    )
    parser.add_argument("--validate", action="store_true", default=False)
    parser.add_argument(
        "--device",
        help="CUDA device (for example cuda:2); defaults to the first Hopper device",
    )
    args = parser.parse_args()

    run_gemm_benchmark(
        args.size,
        args.warmup,
        args.repeat,
        args.iters,
        args.validate,
        args.device,
    )


if __name__ == "__main__":
    main()
