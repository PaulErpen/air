import unittest
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.data.vocabulary import Vocabulary
from .model_knrm import KNRM
from allennlp.common import Tqdm
from allennlp.data.dataloader import PyTorchDataLoader
from ..data_loading import IrTripleDatasetReader, IrLabeledTupleDatasetReader

if __name__ == '__main__':
    unittest.main()


class ModelKNRMTest(unittest.TestCase):
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

    def test_init(self):
        model = KNRM(word_embeddings=self.word_embedder, n_kernels=10)

        self.assertTrue(isinstance(model, KNRM))

    def test_mus(self):
        model = KNRM(word_embeddings=self.word_embedder, n_kernels=10)

        self.assertEquals(model.mu.shape, (1, 1, 1, 10))

    def test_sigmas(self):
        model = KNRM(word_embeddings=self.word_embedder, n_kernels=10)

        self.assertEquals(model.sigma.shape, (1, 1, 1, 10))

    def test_train_batch(self):
        self.assertCountEqual(self.train_batch.keys(), [
                              "query_tokens", "doc_pos_tokens", "doc_neg_tokens"])

    def test_validation_batch(self):
        self.assertCountEqual(self.validation_batch.keys(), [
                              "query_id", "doc_id", "query_tokens", "doc_tokens"])

    def test_validation_batch_shape(self):
        for key in ["query_id", "doc_id"]:
            self.assertEqual(len(self.validation_batch[key]), 32)

    def test_embedding(self):
        model = KNRM(word_embeddings=self.word_embedder, n_kernels=10)

        self.assertEqual(model.word_embeddings(
            self.train_batch["query_tokens"]).shape, (32, 14, 300))

    def test_translation_matrix(self):
        model = KNRM(word_embeddings=self.word_embedder, n_kernels=10)

        self.assertEqual(
            model.create_translation_matrix(
                self.train_batch["query_tokens"], self.train_batch["doc_pos_tokens"]).shape,
            (32, 14, 123)
        )

    def test_gaussian(self):
        model = KNRM(word_embeddings=self.word_embedder, n_kernels=10)
        translation_matrix = model.create_translation_matrix(
            self.train_batch["query_tokens"], self.train_batch["doc_pos_tokens"])

        self.assertEqual(
            model.gaussian(translation_matrix,
                           model.mu[..., 0], model.sigma[..., 0]).shape,
            (32, 14, 123)
        )

    def test_apply_kernel_functions(self):
        model = KNRM(word_embeddings=self.word_embedder, n_kernels=10)
        translation_matrix = model.create_translation_matrix(
            self.train_batch["query_tokens"], self.train_batch["doc_pos_tokens"])

        self.assertEqual(
            model.apply_kernel_functions(translation_matrix).shape,
            (10, 32, 14, 123)
        )

    def test_apply_masking(self):
        model = KNRM(word_embeddings=self.word_embedder, n_kernels=10)
        query = self.train_batch["query_tokens"]
        document = self.train_batch["doc_pos_tokens"]
        translation_matrix = model.create_translation_matrix(query, document)
        kernel_matrix = model.apply_kernel_functions(translation_matrix)

        self.assertEqual(
            model.apply_masking(kernel_matrix, query, document).shape,
            (10, 32, 14, 123)
        )

    def test_apply_sums(self):
        model = KNRM(word_embeddings=self.word_embedder, n_kernels=10)
        query = self.train_batch["query_tokens"]
        document = self.train_batch["doc_pos_tokens"]
        translation_matrix = model.create_translation_matrix(query, document)
        kernel_matrix = model.apply_kernel_functions(translation_matrix)
        masked_kernel_matrix = model.apply_masking(
            kernel_matrix, query, document)

        self.assertEqual(
            model.apply_sums(masked_kernel_matrix).shape,
            (10, 32)
        )

    def test_forward(self):
        model = KNRM(word_embeddings=self.word_embedder, n_kernels=10)
        query = self.train_batch["query_tokens"]
        document = self.train_batch["doc_pos_tokens"]

        self.assertEqual(
            len(model.forward(query, document)),
            32
        )
