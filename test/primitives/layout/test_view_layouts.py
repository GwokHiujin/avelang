#!/usr/bin/env python3
import unittest

import torch

import avelang
import avelang.language as S


@avelang.jit
def kernel_view_layout_row_major(
    input_data: S.Tensor((32,), S.i32),
    output_data: S.Tensor((4, 8), S.i32),
):
    layout_4x8 = S.make_layout((4, 8), (8, 1))
    input_data_cast = S.view(input_data, S.i32, layout_4x8)
    for i in S.range(4):
        for j in S.range(8):
            output_data[i, j] = input_data_cast[i, j]


@avelang.jit
def kernel_view_layout_col_major(
    input_data: S.Tensor((32,), S.i32),
    output_data: S.Tensor((4, 8), S.i32),
):
    layout_4x8 = S.make_layout((4, 8), (1, 4))
    input_data_cast = S.view(input_data, S.i32, layout_4x8)
    for i in S.range(4):
        for j in S.range(8):
            output_data[i, j] = input_data_cast[i, j]


@avelang.jit
def kernel_view_layout_swizzle_1(
    input_data: S.Tensor((32,), S.i32),
    output_data: S.Tensor((4, 2, 4), S.i32),
):
    layout_swizzle = S.make_layout((4, (2, 4)), (2, (1, 8)))
    input_data_cast = S.view(input_data, S.i32, layout_swizzle)
    for i in S.range(4):
        for j in S.range(2):
            for k in S.range(4):
                output_data[i, j, k] = input_data_cast[i, j, k]


@avelang.jit
def kernel_view_layout_swizzle_2(
    input_data: S.Tensor((32,), S.i32),
    output_data: S.Tensor((2, 2, 2, 4), S.i32),
):
    layout_swizzle = S.make_layout(((2, 2), (2, 4)), ((1, 4), (2, 8)))
    input_data_cast = S.view(input_data, S.i32, layout_swizzle)
    for i in S.range(2):
        for j in S.range(2):
            for k in S.range(2):
                for m in S.range(4):
                    output_data[i, j, k, m] = input_data_cast[i, j, k, m]


@avelang.jit
def kernel_view_layout_nested_linear(
    input_data: S.Tensor((8,), S.i32),
    output_data: S.Tensor((8,), S.i32),
):
    layout_swizzle = S.make_layout(((2, 4),), ((4, 1),))
    input_data_cast = S.view(input_data, S.i32, layout_swizzle)
    for t in S.range(8):
        output_data[t] = input_data_cast[t]


class TestViewLayouts(unittest.TestCase):
    def test_view_layout_row_major(self):
        """
        layout((4, 8), (8, 1))
        Expected output:
        [[ 0,  1,  2,  3,  4,  5,  6,  7],
         [ 8,  9, 10, 11, 12, 13, 14, 15],
         [16, 17, 18, 19, 20, 21, 22, 23],
         [24, 25, 26, 27, 28, 29, 30, 31]]
        """
        input_data = torch.arange(32, dtype=torch.int32, device="cuda")
        output_data = torch.zeros((4, 8), dtype=torch.int32, device="cuda")

        expected = input_data.view(4, 8)

        kernel_view_layout_row_major[lambda: ((1, 1, 1), (32, 1, 1))](input_data, output_data)

        actual = output_data.cpu()
        expected = expected.cpu()

        self.assertTrue(
            torch.equal(actual, expected),
            f"Expected: {expected.tolist()}, Actual: {actual.tolist()}",
        )

    def test_view_layout_col_major(self):
        """
        layout((4, 8), (1, 4))
        Expected output:
        [[ 0,  4,  8, 12, 16, 20, 24, 28],
         [ 1,  5,  9, 13, 17, 21, 25, 29],
         [ 2,  6, 10, 14, 18, 22, 26, 30],
         [ 3,  7, 11, 15, 19, 23, 27, 31]]
        """
        input_data = torch.arange(32, dtype=torch.int32, device="cuda")
        output_data = torch.zeros((4, 8), dtype=torch.int32, device="cuda")

        expected = input_data.view(8, 4).t()

        kernel_view_layout_col_major[lambda: ((1, 1, 1), (32, 1, 1))](input_data, output_data)

        actual = output_data.cpu()
        expected = expected.cpu()

        self.assertTrue(
            torch.equal(actual, expected),
            f"Expected: {expected.tolist()}, Actual: {actual.tolist()}",
        )

    def test_view_layout_swizzle_1(self):
        """
        layout((4, (2, 4)), (2, (1, 8)))
        Expected output:
        [[ 0,  1,  8,  9, 16, 17, 24, 25],
         [ 2,  3, 10, 11, 18, 19, 26, 27],
         [ 4,  5, 12, 13, 20, 21, 28, 29],
         [ 6,  7, 14, 15, 22, 23, 30, 31]]
        """
        input_data = torch.arange(32, dtype=torch.int32, device="cuda")
        output_data = torch.zeros((4, 2, 4), dtype=torch.int32, device="cuda")

        kernel_view_layout_swizzle_1[lambda: ((1, 1, 1), (32, 1, 1))](input_data, output_data)

        actual = output_data.cpu()
        expected = input_data.as_strided(size=(4, 2, 4), stride=(2, 1, 8)).cpu()

        self.assertTrue(
            torch.equal(actual, expected),
            f"Expected: {expected.tolist()}, Actual: {actual.tolist()}",
        )

    def test_view_layout_swizzle_2(self):
        """
        layout(((2, 2), (2, 4)), ((1, 4), (2, 8)))
        Expected output:
        [[ 0,  2,  8, 10, 16, 18, 24, 26],
         [ 1,  3,  9, 11, 17, 19, 25, 27],
         [ 4,  6, 12, 14, 20, 22, 28, 30],
         [ 5,  7, 13, 15, 21, 23, 29, 31]]
        """
        input_data = torch.arange(32, dtype=torch.int32, device="cuda")
        output_data = torch.zeros((2, 2, 2, 4), dtype=torch.int32, device="cuda")

        kernel_view_layout_swizzle_2[lambda: ((1, 1, 1), (32, 1, 1))](input_data, output_data)

        actual = output_data.cpu()
        expected = input_data.as_strided(size=(2, 2, 2, 4), stride=(1, 4, 2, 8)).cpu()

        self.assertTrue(
            torch.equal(actual, expected),
            f"Expected: {expected.tolist()}, Actual: {actual.tolist()}",
        )

    def test_view_layout_nested_linear(self):
        """
        layout(((2, 4),), ((4, 1),))
        Linear index into nested dims should map to (i, j) with i fastest.
        Expected output (input_data = 0..7): [0, 4, 1, 5, 2, 6, 3, 7]
        """
        input_data = torch.arange(8, dtype=torch.int32, device="cuda")
        output_data = torch.zeros((8,), dtype=torch.int32, device="cuda")

        kernel_view_layout_nested_linear[lambda: ((1, 1, 1), (32, 1, 1))](input_data, output_data)

        actual = output_data.cpu()
        indices = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], dtype=torch.int64)
        expected = input_data.cpu()[indices]

        self.assertTrue(
            torch.equal(actual, expected),
            f"Expected: {expected.tolist()}, Actual: {actual.tolist()}",
        )


if __name__ == "__main__":
    unittest.main()
