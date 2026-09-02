#!/usr/bin/env python3
import importlib
import math

import pytest
import torch
import torch.nn.functional as F

from avelang_kernels.KDA import fwd


BATCH = 1
SEQUENCE = 8192
HEADS = 64
DIM = 128
LOWER_BOUND = -5.0
VARLEN_SEQ_LENS = (1300, 547, 2048, 963, 271, 3063)


def _hopper_available():
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    return torch.cuda.get_device_capability()[0] >= 9


def _reference_implementation():
    try:
        return importlib.import_module("flash_kda")
    except ImportError:
        pytest.skip("KDA reference package is not installed")


def _make_inputs(seed=100003 + BATCH * 101 + SEQUENCE * 17 + HEADS):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    shape = (BATCH, SEQUENCE, HEADS, DIM)
    q = F.normalize(torch.randn(shape, dtype=torch.float32, device="cuda", generator=generator), p=2, dim=-1).to(
        torch.bfloat16
    )
    k = F.normalize(torch.randn(shape, dtype=torch.float32, device="cuda", generator=generator), p=2, dim=-1).to(
        torch.bfloat16
    )
    v = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    g = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    beta = torch.randn((BATCH, SEQUENCE, HEADS), dtype=torch.bfloat16, device="cuda", generator=generator)
    a_log = torch.rand(HEADS, dtype=torch.float32, device="cuda", generator=generator)
    dt_bias = torch.rand(HEADS, DIM, dtype=torch.float32, device="cuda", generator=generator)
    return q, k, v, g, beta, a_log, dt_bias


@pytest.mark.skipif(not _hopper_available(), reason="NVIDIA Hopper is required")
def test_fixed8192_h64_matches_reference():
    reference = _reference_implementation()
    q, k, v, g, beta, a_log, dt_bias = _make_inputs()
    scale = 1.0 / math.sqrt(DIM)
    actual = torch.empty_like(q)
    expected = torch.empty_like(q)
    kwargs = {
        "A_log": a_log,
        "dt_bias": dt_bias,
        "lower_bound": LOWER_BOUND,
        "initial_state": None,
        "final_state": None,
    }

    reference.fwd(q, k, v, g, beta, scale, expected, **kwargs)
    fwd(q, k, v, g, beta, scale, actual, **kwargs)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.035, atol=0.055)


@pytest.mark.skipif(not _hopper_available(), reason="NVIDIA Hopper is required")
def test_varlen_mix6_h64_matches_reference():
    reference = _reference_implementation()
    q, k, v, g, beta, a_log, dt_bias = _make_inputs()
    scale = 1.0 / math.sqrt(DIM)
    actual = torch.empty_like(q)
    expected = torch.empty_like(q)
    offsets = [0]
    for length in VARLEN_SEQ_LENS:
        offsets.append(offsets[-1] + length)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int64, device="cuda")
    kwargs = {
        "A_log": a_log,
        "dt_bias": dt_bias,
        "lower_bound": LOWER_BOUND,
        "initial_state": None,
        "final_state": None,
        "cu_seqlens": cu_seqlens,
    }

    reference.fwd(q, k, v, g, beta, scale, expected, **kwargs)
    fwd(q, k, v, g, beta, scale, actual, **kwargs)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.035, atol=0.055)
