import unittest
import torch

from .positional_encoding import PositionalEncoding


class PositionalEncodingTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.n_dimensions = 1000
        self.seq_length = 50
        self.some_input = torch.rand(
            self.batch_size, self.seq_length, self.n_dimensions)

    def test_init(self):
        pos = PositionalEncoding(self.n_dimensions)

        self.assertTrue(isinstance(pos, PositionalEncoding))

    def test_forward(self):
        pos = PositionalEncoding(self.n_dimensions)

        self.assertEqual(pos.forward(self.some_input).shape, (32, 50, 1000))
