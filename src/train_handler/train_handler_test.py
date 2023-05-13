import unittest
from .train_handler import TrainHandler
from ..config.config import create_base_config
import os
import torch
import shutil


class TrainHandlerTest(unittest.TestCase):
    def setUp(self):
        self.test_model_dir = "./test_models"
        os.mkdir(self.test_model_dir)

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
            metric_calculator=MockedMetricsCalculator(),
            save_to_disk=False,
            save_models_dir=self.test_model_dir)

        handler.train()

    def test_train_loop_save_to_disk(self):
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
            metric_calculator=MockedMetricsCalculator(),
            save_to_disk=True,
            save_models_dir=self.test_model_dir)

        handler.train()

        models = os.listdir(self.test_model_dir)

        self.assertTrue(len(models) == 10)

    def test_train_loop_early_stopping(self):
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
            metric_calculator=MockedMetricsCalculator(increase=False),
            save_to_disk=True,
            save_models_dir=self.test_model_dir)

        handler.train()

        models = os.listdir(self.test_model_dir)

        self.assertTrue(len(models) == 1)

    def tearDown(self):
        shutil.rmtree(self.test_model_dir)

# ---------------------- MOCKS ---------------------


class MockedMetricsCalculator:
    def __init__(self, increase=True):
        self.calls = 1
        self.increase = increase

    def calculate_metrics(self, a, b):
        if self.increase:
            self.calls = self.calls + 1
        else:
            self.calls = self.calls / 2
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
            k: self.calls for k in metric_keys
        }


class MockedModel:
    def parameters(self):
        return []

    def forward(self, q, d):
        return torch.rand(32)

    def state_dict(self):
        return {
            "parameters": torch.tensor([1, 2, 3, 4, 5])
        }


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
