#!/usr/bin/env python3
import unittest

import torch

import avelang
import avelang.language as S
from avelang.testing import has_rocm


def supports_gfx950() -> bool:
    if not has_rocm():
        return False
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return str(getattr(props, "gcnArchName", "")).split(":", 1)[0] == "gfx950"


@avelang.jit
def kernel_mfma_16x16x32_bf16_f32(
    output: S.Tensor((256,), S.f32),
):
    lane = S.thread_id(0)
    # Two packed BF16 ones per dword, eight values per lane and operand.
    ones = S.full((4,), 0x3F803F80, S.u32)
    accumulator = S.full((4,), 0.0, S.f32)
    accumulator = S.amdgpu.mfma_16x16x32_bf16_f32(
        ones, ones, accumulator
    )
    for index in S.range(4):
        output[lane * 4 + index] = accumulator[index]


@unittest.skipUnless(supports_gfx950(), "Requires a gfx950 ROCm GPU.")
class TestAMDGPUMFMA(unittest.TestCase):
    def test_mfma_16x16x32_bf16_f32(self):
        output = torch.empty((256,), dtype=torch.float32, device="cuda")
        kernel_mfma_16x16x32_bf16_f32[
            lambda: ((1, 1, 1), (64, 1, 1))
        ](output)
        torch.cuda.synchronize()

        expected = torch.full_like(output, 32.0)
        self.assertTrue(
            torch.equal(output, expected),
            msg=(
                "MFMA all-ones dot product did not produce K=32; "
                f"min={output.min().item()} max={output.max().item()}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
