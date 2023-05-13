import unittest
from .config import create_base_config, Config
import os

class ConfigTest(unittest.TestCase):
    def test_base_config(self):
        cfg = create_base_config()
        self.assertTrue(isinstance(cfg, Config))