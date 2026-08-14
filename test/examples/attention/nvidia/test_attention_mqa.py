import unittest

import torch

from avelang_kernels.nvidia_attention import (
    SUPPORTED_SEQUENCES,
    flash_attention_mqa,
)


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
class TestAttentionMQA(unittest.TestCase):
    def setUp(self):
        device_index = get_hopper_device()
        assert device_index is not None
        torch.cuda.set_device(device_index)
        self.device = torch.device(f"cuda:{device_index}")

    def make_inputs(self, sequence):
        torch.manual_seed(7)
        query = torch.randn(
            (16, sequence, 8, 128), dtype=torch.bfloat16, device=self.device
        )
        key = torch.randn(
            (16, sequence, 1, 128), dtype=torch.bfloat16, device=self.device
        )
        value = torch.randn_like(key)
        return query, key, value

    def test_matches_reference(self):
        for sequence in SUPPORTED_SEQUENCES:
            with self.subTest(sequence=sequence):
                query, key, value = self.make_inputs(sequence)
                actual = flash_attention_mqa(query, key, value)

                # Limit the eager reference to 32 query positions so the
                # score tensor remains small enough for routine test runs.
                query_slice = query[0, :32].float().transpose(0, 1)
                key_matrix = key[0, :, 0].float()
                value_matrix = value[0, :, 0].float()
                expected = (
                    torch.softmax(query_slice @ key_matrix.T * (128**-0.5), dim=-1)
                    @ value_matrix
                ).transpose(0, 1)

                torch.testing.assert_close(
                    actual[0, :32].float(), expected, rtol=2e-2, atol=2e-2
                )

    def test_rejects_unsupported_sequence(self):
        query = torch.empty((16, 512, 8, 128), dtype=torch.bfloat16, device=self.device)
        key = torch.empty((16, 512, 1, 128), dtype=torch.bfloat16, device=self.device)
        with self.assertRaisesRegex(ValueError, "sequence length must be one of"):
            flash_attention_mqa(query, key, key)


if __name__ == "__main__":
    unittest.main()
