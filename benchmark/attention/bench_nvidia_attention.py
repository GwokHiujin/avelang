#!/usr/bin/env python3
import argparse
import statistics
import time

import torch

from avelang_kernels.nvidia_attention import (
    BATCH,
    HEAD_DIM,
    QUERY_HEADS,
    SUPPORTED_SEQUENCES,
    flash_attention_mqa,
)


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
            f"NVIDIA attention benchmark requires Hopper or newer, got "
            f"compute capability {major}.{minor}."
        )
    return device


def run_attention_benchmark(
    sequence, warmup, cooldown_ms, repeats, groups, validate, device=None
):
    device = _select_hopper(device)
    if sequence not in SUPPORTED_SEQUENCES:
        raise ValueError(
            f"sequence must be one of {SUPPORTED_SEQUENCES}, got {sequence}."
        )

    query = torch.randn(
        (BATCH, sequence, QUERY_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    key = torch.randn(
        (BATCH, sequence, 1, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    value = torch.randn_like(key)
    output = torch.empty_like(query)

    # Warmup to JIT-compile and stabilize clocks.
    for _ in range(warmup):
        flash_attention_mqa(query, key, value, output=output)
    torch.cuda.synchronize()

    # Match cuda-attn/benchmarks/benchmark.py exactly: time direct launches in
    # independent groups, idle before each group, and report their median.
    samples = []
    for _ in range(groups):
        if cooldown_ms:
            time.sleep(cooldown_ms / 1000.0)
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        for _ in range(repeats):
            flash_attention_mqa(query, key, value, output=output)
        end_evt.record()
        torch.cuda.synchronize()
        samples.append(start_evt.elapsed_time(end_evt) / repeats)
    elapsed_ms = statistics.median(samples)
    elapsed = elapsed_ms * 1.0e-3

    # QK^T and softmax(QK^T)V each perform one multiply and one add.
    flops = 4.0 * BATCH * QUERY_HEADS * sequence * sequence * HEAD_DIM
    tflops = flops / elapsed / 1.0e12
    bytes_moved = (
        BATCH * sequence * QUERY_HEADS * HEAD_DIM * 2
        + 2 * BATCH * sequence * HEAD_DIM * 2
        + BATCH * sequence * QUERY_HEADS * HEAD_DIM * 2
    )
    bandwidth = bytes_moved / elapsed / 1.0e9

    print(
        f"sequence={sequence} time_ms={elapsed_ms:.4f} "
        f"tflops={tflops:.3f} bandwidth_gbs={bandwidth:.3f}"
    )

    if validate:
        # Restrict the eager reference to a small query slice so validation
        # remains practical for the 16K sequence specialization.
        query_positions = min(32, sequence)
        query_slice = query[0, :query_positions].float().transpose(0, 1)
        key_matrix = key[0, :, 0].float()
        value_matrix = value[0, :, 0].float()
        expected = (
            torch.softmax(query_slice @ key_matrix.T * (HEAD_DIM**-0.5), dim=-1)
            @ value_matrix
        ).transpose(0, 1)
        actual = output[0, :query_positions].float()
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        max_abs = torch.max(torch.abs(actual - expected)).item()
        print(f"validation=max_abs_diff:{max_abs:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="NVIDIA Hopper BF16 multi-query attention benchmark"
    )
    parser.add_argument(
        "--sequence", type=int, choices=SUPPORTED_SEQUENCES, default=1024
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--cooldown-ms",
        type=float,
        default=500.0,
        help="untimed GPU idle period before each timing group",
    )
    parser.add_argument("--repeats", "--repeat", type=int, default=10)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--validate", action="store_true", default=False)
    parser.add_argument(
        "--device",
        help="CUDA device (for example cuda:2); defaults to the first Hopper device",
    )
    args = parser.parse_args()

    run_attention_benchmark(
        args.sequence,
        args.warmup,
        args.cooldown_ms,
        args.repeats,
        args.groups,
        args.validate,
        args.device,
    )


if __name__ == "__main__":
    main()
