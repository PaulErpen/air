import unittest
from .model_tk import TK
import unittest
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.data.vocabulary import Vocabulary
from allennlp.common import Tqdm
from allennlp.data.dataloader import PyTorchDataLoader
from ..data_loading import IrTripleDatasetReader, IrLabeledTupleDatasetReader


class ModelTKTest(unittest.TestCase):
    def setUp(self):
        self.vocab = Vocabulary.from_files("./test_data/mock_vocab")
        tokens_embedder = Embedding(vocab=self.vocab,
                                    pretrained_file="./test_data/mini-glove.txt",
                                    embedding_dim=300,
                                    trainable=True,
                                    padding_index=0)
        self.word_embedder = BasicTextFieldEmbedder(
            {"tokens": tokens_embedder})

        _triple_reader = IrTripleDatasetReader(
            lazy=True, max_doc_length=180, max_query_length=30)
        _triple_reader = _triple_reader.read(
            "./test_data/train_triples_test.tsv")
        _triple_reader.index_with(self.vocab)
        self.loader = PyTorchDataLoader(_triple_reader, batch_size=32)
        for b in Tqdm.tqdm(self.loader):
            self.train_batch = b
            break

        _triple_reader = IrLabeledTupleDatasetReader(
            lazy=True, max_doc_length=180, max_query_length=30)
        _triple_reader = _triple_reader.read("./test_data/validation_test.tsv")
        _triple_reader.index_with(self.vocab)
        self.validation_loader = PyTorchDataLoader(
            _triple_reader, batch_size=32)
        for b in Tqdm.tqdm(self.validation_loader):
            self.validation_batch = b
            break

        self.n_kernels = 10
        self.n_layers = 2
        self.n_tf_dim = 40
        self.n_tf_heads = 20

    def test_init(self):
        tk = TK(
            word_embeddings=self.word_embedder,
            n_kernels=self.n_kernels,
            n_layers=self.n_layers,
            n_tf_dim=self.n_tf_dim,
            n_tf_heads=self.n_tf_heads,
        )

        self.assertTrue(isinstance(tk, TK))
