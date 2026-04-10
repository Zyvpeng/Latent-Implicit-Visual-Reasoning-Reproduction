import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from livr.attention_mask import build_livr_attention_mask, build_standard_causal_mask


class TestMask(unittest.TestCase):
    def test_stage1_blocks_prompt_and_answer_to_image(self):
        mask = build_livr_attention_mask(
            seq_len=10,
            image_span=(1, 3),
            prompt_span=(3, 5),
            latent_span=(5, 7),
            answer_span=(7, 10),
            stage="livr_stage1",
            device="cpu",
            dtype=torch.float32,
        )
        blocked = torch.finfo(torch.float32).min
        self.assertEqual(mask[0, 0, 3, 1].item(), blocked)
        self.assertEqual(mask[0, 0, 8, 2].item(), blocked)
        self.assertEqual(mask[0, 0, 5, 1].item(), 0.0)

    def test_stage2_equals_standard_causal(self):
        livr = build_livr_attention_mask(
            seq_len=8,
            image_span=(1, 2),
            prompt_span=(2, 4),
            latent_span=(4, 6),
            answer_span=(6, 8),
            stage="livr_stage2",
            device="cpu",
            dtype=torch.float32,
        )
        standard = build_standard_causal_mask(seq_len=8, device="cpu", dtype=torch.float32)
        self.assertTrue(torch.equal(livr, standard))


if __name__ == "__main__":
    unittest.main()
