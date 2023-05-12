import unittest
from .train_handler import TrainHandler
from ..config.config import create_base_config
import os
import torch
import random


class TrainHandlerTest(unittest.TestCase):
    def setUp(self):
        self.initial_cwd = os.getcwd()
        os.chdir(os.path.join(os.getcwd(), "./src"))
        self.base_config = create_base_config()

    def test_from_config(self):
        handler = TrainHandler.from_config(self.base_config)

        self.assertTrue(isinstance(handler, TrainHandler))

    def test_train_loop(self):
        handler = TrainHandler(
            model_type="mocked",
            model=MockedModel(),
            vocab=None,
            word_embedder=None,
            criterion=mocked_criterion,
            optimizer=MockedOptimizer(),
            qrels=None,
            train_data_loader=MockDataLoader(),
            validation_data_loader=MockDataLoader(),
            metric_calculator=mock_metrics_calculator,
            save_to_disk=False)

        handler.train()

    def tearDown(self):
        os.chdir(self.initial_cwd)

# ---------------------- MOCKS ---------------------


def mock_metrics_calculator(a, b):
    metric_keys = ['MRR@10',  # MRR: Mean Reciprocal Rank, higher is better
                   'Recall@10',
                   'QueriesWithNoRelevant@10',
                   'QueriesWithRelevant@10',
                   'AverageRankGoldLabel@10',
                   'MedianRankGoldLabel@10',
                   'MRR@20',
                   'Recall@20',
                   'QueriesWithNoRelevant@20',
                   'QueriesWithRelevant@20',
                   'AverageRankGoldLabel@20',
                   'MedianRankGoldLabel@20',
                   'MRR@1000',
                   'Recall@1000',
                   'QueriesWithNoRelevant@1000',
                   'QueriesWithRelevant@1000',
                   'AverageRankGoldLabel@1000',
                   'MedianRankGoldLabel@1000',
                   'nDCG@3',  # nDCG: Normalized Discounted Cumulative Gain
                   'nDCG@5',
                   'nDCG@10',
                   'nDCG@20',
                   'nDCG@1000',
                   'QueriesRanked',
                   'MAP@1000']
    return {
        k: random.randint(0, 50) for k in metric_keys
    }


class MockedModel:
    def parameters(self):
        return []

    def forward(self, q, d):
        return torch.rand(32)


class MockedOptimizer:
    def zero_grad(self):
        pass

    def step(self):
        pass


def create_batch():
    return {
        "query_id": torch.rand(32),
        "doc_id": torch.rand(32),
        "query_tokens": torch.rand(32, 50),
        "doc_pos_tokens": torch.rand(32, 50),
        "doc_neg_tokens": torch.rand(32, 50),
        "doc_tokens": torch.rand(32, 50)
    }


class MockDataLoader():
    def __init__(self):
        self.batch_size = 32
        self.data = [create_batch(), create_batch(),
                     create_batch(), create_batch(), create_batch()]
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        element = self.data[self.index]
        self.index += 1
        return element


def mocked_criterion(a, b, c):
    class Loss:
        def backward(self):
            pass
    return Loss()
