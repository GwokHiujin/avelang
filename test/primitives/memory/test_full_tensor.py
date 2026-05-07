#!/usr/bin/env python3
import unittest

import torch

import avelang
import avelang.language as S


@avelang.jit
def kernel_full_1d_i32(
    output_data: S.Tensor((4,), S.i32),
):
    ones = S.full((4,), 1, S.i32)
    for i in S.range(4):
        output_data[i] = ones[i]


@avelang.jit
def kernel_full_2d_f32(
    output_data: S.Tensor((2, 4), S.f32),
):
    acc = S.full((2, 4), 0.0, S.f32)
    for i in S.range(2):
        for j in S.range(4):
            output_data[i, j] = acc[i, j]


@avelang.jit
def kernel_full_vector(
    output_data: S.Tensor((1, 4), S.u32),
):
    ones = S.full((1, 4), 1, S.u32)
    output_data[0] = ones[0]


class TestFullTensor(unittest.TestCase):
    def test_full_1d_i32(self):
        """Test avelang.full with 1D i32 tensor filled with 1s"""
        output_data = torch.zeros((4,), dtype=torch.int32, device="cuda")
        expected = torch.full((4,), 1, dtype=torch.int32, device="cuda")

        kernel_full_1d_i32[lambda: ((1, 1, 1), (1, 1, 1))](output_data)

        actual = output_data.cpu()
        expected = expected.cpu()

        self.assertTrue(
            torch.equal(actual, expected),
            f"Expected: {expected.tolist()}, Actual: {actual.tolist()}",
        )

    def test_full_2d_f32(self):
        """Test avelang.full with 2D f32 tensor filled with 0.0"""
        output_data = torch.zeros((2, 4), dtype=torch.float32, device="cuda")
        expected = torch.zeros((2, 4), dtype=torch.float32, device="cuda")

        kernel_full_2d_f32[lambda: ((1, 1, 1), (1, 1, 1))](output_data)

        actual = output_data.cpu()
        expected = expected.cpu()

        self.assertTrue(
            torch.equal(actual, expected),
            f"Expected: {expected.tolist()}, Actual: {actual.tolist()}",
        )

    def test_full_vector(self):
        """Test avelang.full with 1D vector tensor filled with 1s"""
        output_data = torch.zeros((1, 4), dtype=torch.int32, device="cuda")
        expected = torch.full((1, 4), 1, dtype=torch.int32, device="cuda")

        kernel_full_vector[lambda: ((1, 1, 1), (1, 1, 1))](output_data)

        actual = output_data.cpu()
        expected = expected.cpu()

        self.assertTrue(
            torch.equal(actual, expected),
            f"Expected: {expected.tolist()}, Actual: {actual.tolist()}",
        )


if __name__ == "__main__":
    unittest.main()
