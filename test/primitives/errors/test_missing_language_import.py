from __future__ import annotations

import pytest
import torch

import avelang


@avelang.jit
def kernel_missing_language(x: S.f32):  # noqa: F821
    pass


def test_missing_avelang_language_import_fails():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")

    x = torch.empty((1,), dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError):
        kernel_missing_language[lambda: ((1, 1, 1), (1, 1, 1))](x)
