import unittest

import torch

import avelang
import avelang.language as S
from avelang.testing import has_cuda_nvidia


@avelang.jit
def kernel_atomic_add_i32(
    counter: S.Tensor((1,), S.i32),
    old_values: S.Tensor((256,), S.i32),
):
    index = S.block_id(0) * 64 + S.thread_id(0)
    old = S.nvvm.atomic_add(0, S.convert(1, S.i32), counter)
    old_values[index] = old


@unittest.skipUnless(has_cuda_nvidia(), "Requires CUDA with an NVIDIA GPU.")
class TestNVVMAtomicAdd(unittest.TestCase):
    def test_global_i32_atomic_add(self):
        counter = torch.zeros((1,), dtype=torch.int32, device="cuda")
        old_values = torch.empty((256,), dtype=torch.int32, device="cuda")

        kernel_atomic_add_i32[lambda: ((4, 1, 1), (64, 1, 1))](
            counter, old_values
        )
        torch.cuda.synchronize()

        self.assertEqual(counter.item(), 256)
        expected = torch.arange(256, dtype=torch.int32, device="cuda")
        self.assertTrue(torch.equal(torch.sort(old_values).values, expected))


if __name__ == "__main__":
    unittest.main()
