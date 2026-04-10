import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from livr.latent_tokens import register_row_mask_hook


class DummyEmb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(6, 4))


class TestLatentRowsTrainable(unittest.TestCase):
    def test_only_selected_rows_receive_gradient(self):
        emb = DummyEmb()
        emb.weight.requires_grad_(False)
        register_row_mask_hook(emb, [1, 4])
        loss = emb.weight.sum()
        loss.backward()
        grad = emb.weight.grad
        self.assertTrue(torch.allclose(grad[0], torch.zeros_like(grad[0])))
        self.assertTrue(torch.allclose(grad[2], torch.zeros_like(grad[2])))
        self.assertFalse(torch.allclose(grad[1], torch.zeros_like(grad[1])))
        self.assertFalse(torch.allclose(grad[4], torch.zeros_like(grad[4])))


if __name__ == "__main__":
    unittest.main()
