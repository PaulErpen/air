import unittest
from .config import create_base_config, Config
import os

class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.initial_cwd = os.getcwd()
        os.chdir(os.path.join(os.getcwd(), "./src"))

    def test_base_config(self):
        cfg = create_base_config()
        self.assertTrue(isinstance(cfg, Config))
    
    def tearDown(self):
        os.chdir(self.initial_cwd)
