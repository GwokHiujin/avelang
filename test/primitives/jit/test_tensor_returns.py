#!/usr/bin/env python3
import unittest

import torch

import avelang
import avelang.language as S


@avelang.jit
def make_pair(a: S.u32, b: S.u32) -> S.Tensor((2,), S.u32):
    pair = S.make_local((2,), S.u32)
    pair[0] = a
    pair[1] = b
    return pair


@avelang.jit
def make_pairs(
    a: S.u32, b: S.u32, c: S.u32, d: S.u32
) -> (S.Tensor((2, 3), S.u32), S.Tensor((2, 3), S.u32)):
    first = S.make_local((2, 3), S.u32)
    second = S.make_local((2, 3), S.u32)
    first[0, 0] = a
    first[0, 1] = b
    first[0, 2] = a + b
    first[1, 0] = c
    first[1, 1] = d
    first[1, 2] = c + d
    second[0, 0] = d
    second[0, 1] = c
    second[0, 2] = c + d
    second[1, 0] = b
    second[1, 1] = a
    second[1, 2] = a + b
    return first, second


@avelang.jit
def make_matrix(a: S.u32, b: S.u32, c: S.u32) -> S.Tensor((2, 3), S.u32):
    matrix = S.make_local((2, 3), S.u32)
    matrix[0, 0] = a
    matrix[0, 1] = b
    matrix[0, 2] = c
    matrix[1, 0] = a + c
    matrix[1, 1] = b + c
    matrix[1, 2] = a + b
    return matrix


@avelang.jit
def forward_matrix(a: S.u32, b: S.u32, c: S.u32) -> S.Tensor((2, 3), S.u32):
    return make_matrix(a, b, c)


@avelang.jit
def kernel_tensor_return(
    a: S.Tensor((2,), S.u32),
    b: S.Tensor((2,), S.u32),
    out: S.Tensor((2,), S.u32),
):
    pair = make_pair(a[0] + a[1], b[0] + b[1])
    out[0] = pair[0]
    out[1] = pair[1]


@avelang.jit
def kernel_multiple_tensor_returns(
    a: S.Tensor((2,), S.u32),
    b: S.Tensor((2,), S.u32),
    out: S.Tensor((12,), S.u32),
):
    first, second = make_pairs(a[0], a[1], b[0], b[1])
    out[0] = first[0, 0]
    out[1] = first[0, 1]
    out[2] = first[0, 2]
    out[3] = first[1, 0]
    out[4] = first[1, 1]
    out[5] = first[1, 2]
    out[6] = second[0, 0]
    out[7] = second[0, 1]
    out[8] = second[0, 2]
    out[9] = second[1, 0]
    out[10] = second[1, 1]
    out[11] = second[1, 2]


@avelang.jit
def kernel_multidimensional_tensor_return(
    a: S.Tensor((2,), S.u32),
    b: S.Tensor((2,), S.u32),
    out: S.Tensor((6,), S.u32),
):
    matrix = make_matrix(a[0], a[1], b[0] + b[1])
    out[0] = matrix[0, 0]
    out[1] = matrix[0, 1]
    out[2] = matrix[0, 2]
    out[3] = matrix[1, 0]
    out[4] = matrix[1, 1]
    out[5] = matrix[1, 2]


@avelang.jit
def kernel_forwarded_multidimensional_tensor_return(
    a: S.Tensor((2,), S.u32),
    b: S.Tensor((2,), S.u32),
    out: S.Tensor((6,), S.u32),
):
    matrix = forward_matrix(a[0] + b[0], a[1] + b[1], a[0] + a[1])
    out[0] = matrix[0, 0]
    out[1] = matrix[0, 1]
    out[2] = matrix[0, 2]
    out[3] = matrix[1, 0]
    out[4] = matrix[1, 1]
    out[5] = matrix[1, 2]


class TestTensorReturns(unittest.TestCase):
    def test_tensor_return(self):
        a = torch.tensor([5, 11], dtype=torch.int32, device="cuda")
        b = torch.tensor([7, 13], dtype=torch.int32, device="cuda")
        out = torch.zeros((2,), dtype=torch.int32, device="cuda")

        kernel_tensor_return[lambda: ((1, 1, 1), (1, 1, 1))](a, b, out)

        expected = torch.tensor([16, 20], dtype=torch.int32, device="cuda")
        self.assertTrue(torch.equal(out.cpu(), expected.cpu()))

    def test_multiple_tensor_returns(self):
        a = torch.tensor([5, 11], dtype=torch.int32, device="cuda")
        b = torch.tensor([7, 13], dtype=torch.int32, device="cuda")
        out = torch.zeros((12,), dtype=torch.int32, device="cuda")

        kernel_multiple_tensor_returns[lambda: ((1, 1, 1), (1, 1, 1))](
            a, b, out
        )

        expected = torch.tensor(
            [5, 11, 16, 7, 13, 20, 13, 7, 20, 11, 5, 16],
            dtype=torch.int32,
            device="cuda",
        )
        self.assertTrue(torch.equal(out.cpu(), expected.cpu()))

    def test_multidimensional_tensor_return(self):
        a = torch.tensor([5, 11], dtype=torch.int32, device="cuda")
        b = torch.tensor([7, 13], dtype=torch.int32, device="cuda")
        out = torch.zeros((6,), dtype=torch.int32, device="cuda")

        kernel_multidimensional_tensor_return[
            lambda: ((1, 1, 1), (1, 1, 1))
        ](a, b, out)

        expected = torch.tensor(
            [5, 11, 20, 25, 31, 16], dtype=torch.int32, device="cuda"
        )
        self.assertTrue(torch.equal(out.cpu(), expected.cpu()))

    def test_forwarded_multidimensional_tensor_return(self):
        a = torch.tensor([5, 11], dtype=torch.int32, device="cuda")
        b = torch.tensor([7, 13], dtype=torch.int32, device="cuda")
        out = torch.zeros((6,), dtype=torch.int32, device="cuda")

        kernel_forwarded_multidimensional_tensor_return[
            lambda: ((1, 1, 1), (1, 1, 1))
        ](a, b, out)

        expected = torch.tensor(
            [12, 24, 16, 28, 40, 36], dtype=torch.int32, device="cuda"
        )
        self.assertTrue(torch.equal(out.cpu(), expected.cpu()))


if __name__ == "__main__":
    unittest.main()
