#!/usr/bin/env python3
import unittest

import torch

import avelang
import avelang.language as S
from avelang import testing


NUMEL = 4096
THREADS = 256
BLOCKS = NUMEL // THREADS
L2E = 1.442695041
SMALL_THRESHOLD = 4.997253418e-3


@avelang.jit
def fast_tanhf(a: S.f32) -> S.f32:
    zero = S.convert(0.0, S.f32)
    one = S.convert(1.0, S.f32)
    neg_two_l2e = S.convert(-2.0 * L2E, S.f32)
    small = S.convert(SMALL_THRESHOLD, S.f32)

    s = S.abs(a)

    e = S.exp2(neg_two_l2e * s)
    r = S.amdgpu.rcp(e + one)
    r = r - e * r

    r = -r if a < zero else r
    return a if s < small else r


@avelang.jit
def kernel_fast_tanh_approx(
    inp: S.Tensor((NUMEL,), S.f32),
    out: S.Tensor((NUMEL,), S.f32),
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    out[idx] = fast_tanhf(inp[idx])


def make_test_input() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    sweep = torch.linspace(-10.0, 10.0, 2048, dtype=torch.float32)
    random_values = torch.randn(2017, generator=generator, dtype=torch.float32) * 3.0
    threshold_cases = torch.tensor(
        [
            -20.0,
            -10.0,
            -5.0,
            -1.0,
            -SMALL_THRESHOLD,
            -SMALL_THRESHOLD / 2.0,
            0.0,
            SMALL_THRESHOLD / 2.0,
            SMALL_THRESHOLD,
            1.0,
            5.0,
            10.0,
            20.0,
            -1.0e-7,
            1.0e-7,
            -1.0e-4,
            1.0e-4,
            -1.0e-2,
            1.0e-2,
            -0.5,
            0.5,
            -2.0,
            2.0,
            -8.0,
            8.0,
            -12.0,
            12.0,
            -15.0,
            15.0,
            -30.0,
            30.0,
        ],
        dtype=torch.float32,
    )
    values = torch.cat((sweep, random_values, threshold_cases))
    assert values.numel() == NUMEL
    return values


class TestTanhFastApprox(unittest.TestCase):
    def setUp(self):
        if not testing.has_rocm():
            self.skipTest("ROCm/HIP not available.")

    def test_fast_tanh_stays_close_to_torch(self):
        inp = make_test_input().to("cuda")
        out = torch.empty_like(inp)

        kernel_fast_tanh_approx[lambda: ((BLOCKS, 1, 1), (THREADS, 1, 1))](inp, out)

        expected = torch.tanh(inp).cpu()
        actual = out.cpu()
        abs_diff = (actual - expected).abs()
        rel_diff = abs_diff / expected.abs().clamp_min(1.0e-20)

        max_abs = abs_diff.max().item()
        max_rel = rel_diff.max().item()
        max_abs_idx = abs_diff.argmax().item()
        max_rel_idx = rel_diff.argmax().item()

        self.assertLessEqual(
            max_abs,
            2.0e-5,
            msg=(
                f"max_abs={max_abs} at x={inp[max_abs_idx].item()} "
                f"actual={actual[max_abs_idx].item()} "
                f"expected={expected[max_abs_idx].item()}"
            ),
        )
        self.assertLessEqual(
            max_rel,
            2.0e-5,
            msg=(
                f"max_rel={max_rel} at x={inp[max_rel_idx].item()} "
                f"actual={actual[max_rel_idx].item()} "
                f"expected={expected[max_rel_idx].item()}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
