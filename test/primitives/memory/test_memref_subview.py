#!/usr/bin/env python3
import unittest

import torch

import avelang
import avelang.language as S


@avelang.jit
def kernel_memref_subview(origin_data: S.Tensor((8, 8), S.i32)):
    a = 1
    b = 1
    subviewd = S.subview(origin_data, (a, b), (4, 4), (2, 2))
    for i in S.range(4):
        for j in S.range(4):
            subviewd[i, j] = 999


class TestMemrefSubviewOps(unittest.TestCase):
    """Unit test for verifying that modifying a subview affects the original memref."""

    def setUp(self):
        self.origin_data = torch.arange(64, dtype=torch.int32, device="cuda").reshape(8, 8)

    def test_subview_modification_reflects_on_origin(self):
        """Check that subview modification propagates to the original tensor."""
        kernel_memref_subview[lambda: ((1, 1, 1), (32, 1, 1))](self.origin_data)

        expected = torch.full((4, 4), 999, dtype=torch.int32, device="cpu")
        actual = self.origin_data[1:8:2, 1:8:2][:4, :4].to("cpu")

        self.assertTrue(
            torch.allclose(actual, expected, rtol=1e-2, atol=1e-2),
            msg=(
                "Subview tensor modification did not reflect in the original memref.\n"
                f"Expected:\n{expected}\nActual:\n{actual}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
