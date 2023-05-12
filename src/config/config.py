import os
from enum import Enum


class Config:
    def __init__(self,
                 vocab_directory: str,
                 pre_trained_embedding: str,
                 model: str,
                 train_data: str,
                 validation_data: str,
                 test_data: str,
                 qurels: str):
        self.vocab_directory: str = vocab_directory
        self.pre_trained_embedding: str = pre_trained_embedding
        self.model: ModelType = model
        self.train_data: str = train_data
        self.validation_data: str = validation_data
        self.test_data: str = test_data
        self.qurels: str = qurels
        self.validate()

    def validate(self):
        if not os.path.exists(self.vocab_directory):
            raise Exception(
                "Configuration-Error: vocab_directory does not exist!")
        if not os.path.isdir(self.vocab_directory):
            raise Exception(
                "Configuration-Error: vocab_directory is not a directory!")

        if not os.path.exists(self.pre_trained_embedding):
            raise Exception(
                "Configuration-Error: pre_trained_embedding does not exist!")
        if not os.path.isfile(self.pre_trained_embedding):
            raise Exception(
                "Configuration-Error: pre_trained_embedding is not a file!")

        if not (self.model == "knrm" or self.model == "tk"):
            raise Exception(
                "Configuration-Error: model type invalid!")

        if not os.path.exists(self.train_data):
            raise Exception("Configuration-Error: train_data does not exist!")
        if not os.path.isfile(self.train_data):
            raise Exception("Configuration-Error: train_data is not a file!")

        if not os.path.exists(self.test_data):
            raise Exception("Configuration-Error: test_data does not exist!")
        if not os.path.isfile(self.test_data):
            raise Exception("Configuration-Error: test_data is not a file!")

        if not os.path.exists(self.validation_data):
            raise Exception(
                "Configuration-Error: validation_data does not exist!")
        if not os.path.isfile(self.validation_data):
            raise Exception(
                "Configuration-Error: validation_data is not a file!")

        if not os.path.exists(self.qurels):
            raise Exception("Configuration-Error: qurels does not exist!")
        if not os.path.isfile(self.qurels):
            raise Exception("Configuration-Error: qurels is not a file!")


def create_base_config():
    return Config(
        vocab_directory="../data/Part-2/allen_vocab_lower_10",
        pre_trained_embedding="../data/Part-2/glove.42B.300d.txt",
        model="knrm",
        train_data="../data/Part-2/triples.train.tsv",
        validation_data="../data/Part-2/msmarco_tuples.validation.tsv",
        test_data="../data/Part-2/msmarco_tuples.test.tsv",
        qurels="../data/Part-2/msmarco_qrels.txt",
    )
