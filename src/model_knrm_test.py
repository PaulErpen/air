import unittest
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.data.vocabulary import Vocabulary
from .model_knrm import KNRM
import torch
from allennlp.common import Tqdm
from allennlp.data.dataloader import PyTorchDataLoader,PyTorchDataLoader
import torch
from .data_loading import IrTripleDatasetReader

if __name__ == '__main__':
    unittest.main()

class ModelKNRMTest(unittest.TestCase):
    def setUp(self):
        self.vocab = Vocabulary.from_files("./test_data/mock_vocab")
        tokens_embedder = Embedding(vocab=self.vocab,
                           pretrained_file= "./test_data/mini-glove.txt",
                           embedding_dim=300,
                           trainable=True,
                           padding_index=0) # type: ignore
        self.word_embedder = BasicTextFieldEmbedder({"tokens": tokens_embedder}) # type: ignore

        _triple_reader = IrTripleDatasetReader(lazy=True, max_doc_length=180, max_query_length=30)
        _triple_reader = _triple_reader.read("./test_data/train_triples_test.tsv")
        _triple_reader.index_with(self.vocab)
        self.loader = PyTorchDataLoader(_triple_reader, batch_size=32)
        for b in Tqdm.tqdm(self.loader):
            self.batch = b
            break

    def test_init(self):
        model = KNRM(word_embeddings=self.word_embedder, n_kernels=10) # type: ignore

        self.assertTrue(isinstance(model, KNRM))

    def test_mus(self):
        model = KNRM(word_embeddings=self.word_embedder, n_kernels=10) # type: ignore

        self.assertEquals(model.mu.shape, (1, 1, 1, 10))
    
    def test_sigmas(self):
        model = KNRM(word_embeddings=self.word_embedder, n_kernels=10) # type: ignore

        self.assertEquals(model.sigma.shape, (1, 1, 1, 10))

    def test_batch(self):
        self.assertCountEqual(self.batch.keys(), ["query_tokens", "doc_pos_tokens", "doc_neg_tokens"])
    
    def test_embedding(self):
        model = KNRM(word_embeddings=self.word_embedder, n_kernels=10) # type: ignore

        self.assertEqual(model.word_embeddings(self.batch["query_tokens"]).shape, (32, 14, 300))
    
    def test_translation_matrix(self):
        model = KNRM(word_embeddings=self.word_embedder, n_kernels=10) # type: ignore

        self.assertEqual(
            model.create_translation_matrix(self.batch["query_tokens"], self.batch["doc_pos_tokens"]).shape, # type: ignore
            (32, 14, 123)
        )

    def test_gaussian(self):
        model = KNRM(word_embeddings=self.word_embedder, n_kernels=10) # type: ignore
        translation_matrix = model.create_translation_matrix(self.batch["query_tokens"], self.batch["doc_pos_tokens"]) # type: ignore

        self.assertEqual(
            model.gaussian(translation_matrix, model.mu[..., 0], model.sigma[..., 0]).shape,
            (32, 14, 123)
        )
    
    def test_apply_kernel_functions(self):
        model = KNRM(word_embeddings=self.word_embedder, n_kernels=10) # type: ignore
        translation_matrix = model.create_translation_matrix(self.batch["query_tokens"], self.batch["doc_pos_tokens"])

        self.assertEqual(
            model.apply_kernel_functions(translation_matrix).shape,
            (10, 32, 14, 123)
        )
    
    def test_apply_masking(self):
        model = KNRM(word_embeddings=self.word_embedder, n_kernels=10) # type: ignore
        query = self.batch["query_tokens"]
        document = self.batch["doc_pos_tokens"]
        translation_matrix = model.create_translation_matrix(query, document)
        kernel_matrix = model.apply_kernel_functions(translation_matrix)

        self.assertEqual(
            model.apply_masking(kernel_matrix, query, document).shape,
            (10, 32, 14, 123)
        )
    
    def test_apply_sums(self):
        model = KNRM(word_embeddings=self.word_embedder, n_kernels=10) # type: ignore
        query = self.batch["query_tokens"]
        document = self.batch["doc_pos_tokens"]
        translation_matrix = model.create_translation_matrix(query, document)
        kernel_matrix = model.apply_kernel_functions(translation_matrix)
        masked_kernel_matrix = model.apply_masking(kernel_matrix, query, document)

        self.assertEqual(
            model.apply_sums(masked_kernel_matrix).shape,
            (10, 32)
        )