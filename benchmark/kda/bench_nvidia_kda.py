#!/usr/bin/env python3
"""Benchmark Avelang's fixed-shape Hopper KDA kernel."""

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from avelang_kernels.KDA import fwd as avelang_fwd


BATCH = 1
SEQUENCE = 8192
HEADS = 64
DIM = 128
CHUNK = 16
LOWER_BOUND = -5.0


@dataclass
class PoolEntry:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    out: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor


class RandomPool:
    """Preallocate independent inputs for full-forward timing."""

    def __init__(self, pool_size, seed, device):
        self.scale = 1.0 / math.sqrt(DIM)
        self.entries = []
        for i in range(pool_size):
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seed) + 1009 * i + 9173 * HEADS + SEQUENCE)
            shape = (BATCH, SEQUENCE, HEADS, DIM)
            q = F.normalize(
                torch.randn(shape, dtype=torch.float32, device=device, generator=gen),
                p=2,
                dim=-1,
            ).to(torch.bfloat16)
            k = F.normalize(
                torch.randn(shape, dtype=torch.float32, device=device, generator=gen),
                p=2,
                dim=-1,
            ).to(torch.bfloat16)
            v = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=gen)
            g = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=gen)
            beta = torch.randn(
                (BATCH, SEQUENCE, HEADS),
                dtype=torch.bfloat16,
                device=device,
                generator=gen,
            )
            out = torch.empty_like(q)
            a_log = torch.rand(HEADS, dtype=torch.float32, device=device, generator=gen)
            dt_bias = torch.rand(HEADS, DIM, dtype=torch.float32, device=device, generator=gen)
            self.entries.append(PoolEntry(q, k, v, g, beta, out, a_log, dt_bias))

        for field in PoolEntry.__dataclass_fields__:
            pointers = [getattr(entry, field).data_ptr() for entry in self.entries]
            if len(pointers) != len(set(pointers)):
                raise RuntimeError(f"random pool reuses storage for {field}")


def _ensure_hopper_available():
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("NVIDIA CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        raise RuntimeError(f"KDA benchmark requires Hopper or newer, got {major}.{minor}")


def _call(module_fwd, pool, entry):
    module_fwd(
        entry.q,
        entry.k,
        entry.v,
        entry.g,
        entry.beta,
        pool.scale,
        entry.out,
        A_log=entry.A_log,
        dt_bias=entry.dt_bias,
        lower_bound=LOWER_BOUND,
        initial_state=None,
        final_state=None,
    )


def _bench(module_fwd, pool, warmup, iters, repeats):
    idx = 0
    for _ in range(max(1, warmup)):
        _call(module_fwd, pool, pool.entries[idx % len(pool.entries)])
        idx += 1
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            entry = pool.entries[idx % len(pool.entries)]
            starts[i].record()
            _call(module_fwd, pool, entry)
            ends[i].record()
            idx += 1
        torch.cuda.synchronize()
        samples.extend(float(start.elapsed_time(end)) for start, end in zip(starts, ends))
    samples.sort()
    n = len(samples)
    return {
        "mean_ms": sum(samples) / n,
        "median_ms": samples[n // 2],
        "min_ms": samples[0],
        "max_ms": samples[-1],
    }


def _validate_first_tile(pool):
    entry = pool.entries[0]
    _call(avelang_fwd, pool, entry)
    torch.cuda.synchronize()

    q = F.normalize(entry.q[0, :CHUNK, 0].float(), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(entry.k[0, :CHUNK, 0].float(), p=2, dim=-1).to(torch.bfloat16)
    v = entry.v[0, :CHUNK, 0]
    gate = LOWER_BOUND * torch.sigmoid(torch.exp(entry.A_log[0]) * (entry.g[0, :CHUNK, 0].float() + entry.dt_bias[0]))
    gate_cumsum = gate.cumsum(dim=0)
    k_decayed = k * torch.exp(gate_cumsum).to(torch.bfloat16)
    q_decayed = (
        q * torch.exp(gate_cumsum).to(torch.bfloat16) * torch.tensor(pool.scale, dtype=torch.bfloat16, device=q.device)
    )
    k_inverse = k * torch.exp(-gate_cumsum).to(torch.bfloat16)

    beta = torch.sigmoid(entry.beta[0, :CHUNK, 0].float())
    transition = (k_decayed @ k_inverse.T).to(torch.float16)
    transition = torch.tril(transition, diagonal=-1) * beta.to(torch.float16).unsqueeze(-1)
    inverse = torch.eye(CHUNK, dtype=torch.float16, device=q.device) - transition
    transition2 = transition @ transition
    inverse = inverse + inverse @ transition2
    transition4 = transition2 @ transition2
    inverse = inverse + inverse @ transition4
    transition8 = transition4 @ transition4
    inverse = (inverse + inverse @ transition8).to(torch.bfloat16)

    update = inverse @ (v * beta.to(torch.bfloat16).unsqueeze(-1))
    expected = torch.tril(q_decayed @ k_inverse.T) @ update
    actual = entry.out[0, :CHUNK, 0]
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.035, atol=0.055)
    max_abs = torch.max(torch.abs(actual.float() - expected.float())).item()
    print(f"validation=max_abs_diff:{max_abs:.6f}")


def run_kda_benchmark(pool_size, warmup, iters, repeats, seed, validate):
    if pool_size < 2:
        raise ValueError("pool_size must be at least 2 for random-pool benchmarking")
    _ensure_hopper_available()
    pool = RandomPool(pool_size, seed, torch.device("cuda"))
    result = _bench(avelang_fwd, pool, warmup, iters, repeats)
    print(
        f"B={BATCH} T={SEQUENCE} H={HEADS} D={DIM} "
        f"pool_size={pool_size} mean_ms={result['mean_ms']:.4f} "
        f"median_ms={result['median_ms']:.4f} "
        f"min_ms={result['min_ms']:.4f} max_ms={result['max_ms']:.4f}"
    )
    if validate:
        _validate_first_tile(pool)


def main():
    parser = argparse.ArgumentParser(description="NVIDIA Hopper BF16 fixed-shape KDA benchmark")
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--validate", action="store_true", default=False)
    args = parser.parse_args()

    run_kda_benchmark(
        args.pool_size,
        args.warmup,
        args.iters,
        args.repeats,
        args.seed,
        args.validate,
    )


if __name__ == "__main__":
    main()
