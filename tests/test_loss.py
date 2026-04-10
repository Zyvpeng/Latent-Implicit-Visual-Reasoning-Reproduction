import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from livr.data import build_labels


class TestLossMask(unittest.TestCase):
    def test_only_answer_tokens_have_labels(self):
        input_ids = torch.tensor([10, 11, 12, 13, 14, 15])
        labels = build_labels(input_ids=input_ids, answer_span=(4, 6))
        self.assertTrue(torch.equal(labels, torch.tensor([-100, -100, -100, -100, 14, 15])))


if __name__ == "__main__":
    unittest.main()
