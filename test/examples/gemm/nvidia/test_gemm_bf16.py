#!/usr/bin/env python3
import unittest

import torch

from avelang_kernels.nvidia_gemm import SUPPORTED_SIZES, gemm_bf16


def get_hopper_device():
    if not torch.cuda.is_available():
        return None
    for device_index in range(torch.cuda.device_count()):
        major, _minor = torch.cuda.get_device_capability(device_index)
        if major >= 9:
            return device_index
    return None


@unittest.skipUnless(
    get_hopper_device() is not None,
    "Requires an NVIDIA Hopper-or-newer GPU.",
)
class TestNvidiaGemmBf16(unittest.TestCase):
    def setUp(self):
        device_index = get_hopper_device()
        assert device_index is not None
        torch.cuda.set_device(device_index)
        self.device = torch.device(f"cuda:{device_index}")

    def test_random_1024_matches_torch(self):
        torch.manual_seed(17)
        a = torch.randn(
            (1024, 1024), dtype=torch.bfloat16, device=self.device
        )
        b = torch.randn_like(a)

        actual = gemm_bf16(a, b)
        expected = a @ b.T

        self.assertTrue(torch.equal(actual, expected))

    def test_all_supported_sizes(self):
        # Ones provide an exact BF16 oracle for these power-of-two K sizes and
        # keep the 16384 case from allocating another 512 MiB reference.
        for size in SUPPORTED_SIZES:
            with self.subTest(size=size):
                a = torch.ones(
                    (size, size), dtype=torch.bfloat16, device=self.device
                )
                b = torch.ones_like(a)
                out = torch.empty_like(a)

                result = gemm_bf16(a, b, out=out)

                self.assertIs(result, out)
                self.assertTrue(torch.all(result == size).item())
                del a, b, out, result

    def test_rejects_unsupported_size(self):
        a = torch.empty(
            (512, 512), dtype=torch.bfloat16, device=self.device
        )
        with self.assertRaisesRegex(ValueError, "size must be one of"):
            gemm_bf16(a, a)


if __name__ == "__main__":
    unittest.main()
