#!/usr/bin/env python3
import unittest

import torch
from avelang_kernels.nvidia_moe import (
    SUPPORTED_SIZES,
    moe_gemm_fp8,
    silu_mul_quant_fp8,
)


def get_hopper_device():
    if not torch.cuda.is_available():
        return None
    for device_index in range(torch.cuda.device_count()):
        major, _minor = torch.cuda.get_device_capability(device_index)
        if major >= 9:
            return device_index
    return None


def dequantize_weight(
    weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    rows, columns = weight.shape
    row_blocks = torch.arange(rows, device=weight.device) // 128
    column_blocks = torch.arange(columns, device=weight.device) // 128
    return weight.float() * scale[
        row_blocks[:, None], column_blocks[None, :]
    ]


@unittest.skipUnless(
    get_hopper_device() is not None,
    "Requires an NVIDIA Hopper-or-newer GPU.",
)
class TestNvidiaMoeFp8(unittest.TestCase):
    def setUp(self):
        device_index = get_hopper_device()
        assert device_index is not None
        torch.cuda.set_device(device_index)
        self.device = torch.device(f"cuda:{device_index}")

    def test_random_1024_multiple_experts_matches_torch(self):
        torch.manual_seed(7)
        size = 1024
        num_experts = 3
        scale_blocks = size // 128
        a = (
            torch.randn((size, size), device=self.device)
            .mul_(0.5)
            .to(torch.float8_e4m3fn)
        )
        b = (
            torch.randn(
                (num_experts, size, size), device=self.device
            )
            .mul_(0.5)
            .to(torch.float8_e4m3fn)
        )
        a_scale = torch.empty(
            (scale_blocks, size), device=self.device
        ).uniform_(0.015, 0.025)
        b_scale = torch.empty(
            (num_experts, scale_blocks, scale_blocks),
            device=self.device,
        ).uniform_(0.015, 0.025)
        # Blocks 2/3 and 4/5 deliberately select different experts.  This
        # covers both the clustered B multicast and per-CTA fallback paths.
        block_expert_ids = torch.tensor(
            [0, 0, 1, 2, 2, 1, 0, 2],
            dtype=torch.int32,
            device=self.device,
        )

        actual = moe_gemm_fp8(
            a, b, a_scale, b_scale, block_expert_ids
        )

        k_blocks = torch.arange(size, device=self.device) // 128
        dequantized_a = a.float() * a_scale.T[:, k_blocks]
        expected = torch.empty(
            (size, size), dtype=torch.float32, device=self.device
        )
        for block, expert in enumerate(block_expert_ids.tolist()):
            rows = slice(block * 128, (block + 1) * 128)
            expected[rows] = dequantized_a[rows] @ dequantize_weight(
                b[expert], b_scale[expert]
            ).T

        torch.testing.assert_close(
            actual.float(), expected, rtol=0.05, atol=0.005
        )

    def test_all_supported_square_sizes(self):
        # Ones and unit scales are exactly representable in every input and
        # output type, and avoid a second 512 MiB oracle at size 16384.
        for size in SUPPORTED_SIZES:
            with self.subTest(size=size):
                scale_blocks = size // 128
                a = torch.ones(
                    (size, size),
                    dtype=torch.float8_e4m3fn,
                    device=self.device,
                )
                b = a.unsqueeze(0).contiguous()
                a_scale = torch.ones(
                    (scale_blocks, size),
                    dtype=torch.float32,
                    device=self.device,
                )
                b_scale = torch.ones(
                    (1, scale_blocks, scale_blocks),
                    dtype=torch.float32,
                    device=self.device,
                )
                block_expert_ids = torch.zeros(
                    (size // 128,),
                    dtype=torch.int32,
                    device=self.device,
                )
                out = torch.empty(
                    (size, size),
                    dtype=torch.bfloat16,
                    device=self.device,
                )

                result = moe_gemm_fp8(
                    a,
                    b,
                    a_scale,
                    b_scale,
                    block_expert_ids,
                    out=out,
                )

                self.assertIs(result, out)
                self.assertTrue(torch.all(result == size).item())
                del a, b, a_scale, b_scale, block_expert_ids, out, result
                torch.cuda.empty_cache()

    def test_rectangular_shape_matches_torch(self):
        torch.manual_seed(19)
        rows, n, k, num_experts = 256, 256, 128, 3
        a = torch.randn(
            (rows, k), dtype=torch.float32, device=self.device
        ).to(torch.float8_e4m3fn)
        b = torch.randn(
            (num_experts, n, k),
            dtype=torch.float32,
            device=self.device,
        ).to(torch.float8_e4m3fn)
        a_scale = torch.empty((1, rows), device=self.device).uniform_(
            0.015, 0.025
        )
        b_scale = torch.empty(
            (num_experts, 2, 1), device=self.device
        ).uniform_(0.015, 0.025)
        block_expert_ids = torch.tensor(
            [0, 2], dtype=torch.int32, device=self.device
        )

        actual = moe_gemm_fp8(
            a, b, a_scale, b_scale, block_expert_ids
        )
        expected = torch.empty(
            (rows, n), dtype=torch.float32, device=self.device
        )
        for block, expert in enumerate(block_expert_ids.tolist()):
            block_rows = slice(block * 128, (block + 1) * 128)
            expected[block_rows] = (
                a[block_rows].float()
                * a_scale[:, block_rows].T
            ) @ dequantize_weight(b[expert], b_scale[expert]).T
        torch.testing.assert_close(
            actual.float(), expected, rtol=0.05, atol=0.005
        )

    def test_silu_mul_quant_matches_torch(self):
        torch.manual_seed(23)
        rows, inter_dim = 256, 128
        input = torch.randn(
            (rows, 2 * inter_dim),
            dtype=torch.bfloat16,
            device=self.device,
        ).mul_(0.2)
        output = torch.empty(
            (rows, inter_dim),
            dtype=torch.float8_e4m3fn,
            device=self.device,
        )
        scale = torch.empty(
            (1, rows), dtype=torch.float32, device=self.device
        )

        silu_mul_quant_fp8(input, output, scale)

        gate, up = input.float().chunk(2, dim=1)
        expected = torch.nn.functional.silu(gate) * up
        actual = output.float() * scale.T
        torch.testing.assert_close(actual, expected, rtol=0.08, atol=0.008)

    def test_rejects_unclustered_rows(self):
        a = torch.empty(
            (128, 128),
            dtype=torch.float8_e4m3fn,
            device=self.device,
        )
        with self.assertRaisesRegex(ValueError, "rows must be divisible by 256"):
            moe_gemm_fp8(a, a.unsqueeze(0), a.float(), a.float(), a)


if __name__ == "__main__":
    unittest.main()
