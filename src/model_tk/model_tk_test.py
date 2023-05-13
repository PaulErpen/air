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

    def test_positional_encoding(self):
        tk = TK(
            word_embeddings=self.word_embedder,
            n_kernels=self.n_kernels,
            n_layers=self.n_layers,
            n_tf_dim=self.n_tf_dim,
            n_tf_heads=self.n_tf_heads,
        )

        query_embeddings = tk.word_embeddings(self.train_batch["query_tokens"])

        self.assertEqual(
            tk.positional_encoding(query_embeddings).shape,
            query_embeddings.shape
        )

    def test_transformer(self):
        tk = TK(
            word_embeddings=self.word_embedder,
            n_kernels=self.n_kernels,
            n_layers=self.n_layers,
            n_tf_dim=self.n_tf_dim,
            n_tf_heads=self.n_tf_heads,
        )

        query_embeddings = tk.word_embeddings(self.train_batch["query_tokens"])
        contextualized_query_embeddings = tk.positional_encoding(
            query_embeddings)

        self.assertEqual(
            tk.transformer(contextualized_query_embeddings).shape,
            (32, 14, 300)
        )

    def test_translation_matrix(self):
        tk = TK(
            word_embeddings=self.word_embedder,
            n_kernels=self.n_kernels,
            n_layers=self.n_layers,
            n_tf_dim=self.n_tf_dim,
            n_tf_heads=self.n_tf_heads,
        )
        query_embeddings = tk.word_embeddings(self.train_batch["query_tokens"])
        document_embeddings = tk.word_embeddings(
            self.train_batch["doc_pos_tokens"])
        contextualized_query_embeddings = tk.positional_encoding(
            query_embeddings)
        contextualized_document_embeddings = tk.positional_encoding(
            document_embeddings)
        transformed_query_embeddings = tk.transformer(
            contextualized_query_embeddings)
        transformed_document_embeddings = tk.transformer(
            contextualized_document_embeddings)

        self.assertEquals(
            tk.create_translation_matrix(
                transformed_query_embeddings, transformed_document_embeddings).shape,
            (32, 14, 123)
        )

    def test_kernel_matrix(self):
        tk = TK(
            word_embeddings=self.word_embedder,
            n_kernels=self.n_kernels,
            n_layers=self.n_layers,
            n_tf_dim=self.n_tf_dim,
            n_tf_heads=self.n_tf_heads,
        )
        query_embeddings = tk.word_embeddings(self.train_batch["query_tokens"])
        document_embeddings = tk.word_embeddings(
            self.train_batch["doc_pos_tokens"])
        contextualized_query_embeddings = tk.positional_encoding(
            query_embeddings)
        contextualized_document_embeddings = tk.positional_encoding(
            document_embeddings)
        transformed_query_embeddings = tk.transformer(
            contextualized_query_embeddings)
        transformed_document_embeddings = tk.transformer(
            contextualized_document_embeddings)
        translation_matrix = tk.create_translation_matrix(
            transformed_query_embeddings, transformed_document_embeddings)

        self.assertEquals(
            tk.apply_kernel_functions(translation_matrix).shape,
            (self.n_kernels, 32, 14, 123)
        )

    def test_masked_kernel_matrix(self):
        tk = TK(
            word_embeddings=self.word_embedder,
            n_kernels=self.n_kernels,
            n_layers=self.n_layers,
            n_tf_dim=self.n_tf_dim,
            n_tf_heads=self.n_tf_heads,
        )
        query_embeddings = tk.word_embeddings(self.train_batch["query_tokens"])
        document_embeddings = tk.word_embeddings(
            self.train_batch["doc_pos_tokens"])
        contextualized_query_embeddings = tk.positional_encoding(
            query_embeddings)
        contextualized_document_embeddings = tk.positional_encoding(
            document_embeddings)
        transformed_query_embeddings = tk.transformer(
            contextualized_query_embeddings)
        transformed_document_embeddings = tk.transformer(
            contextualized_document_embeddings)
        translation_matrix = tk.create_translation_matrix(
            transformed_query_embeddings, transformed_document_embeddings)
        kernel_matrix = tk.apply_kernel_functions(translation_matrix)

        self.assertEquals(
            tk.apply_masking(
                kernel_matrix, self.train_batch["query_tokens"], self.train_batch["doc_pos_tokens"]).shape,
            (self.n_kernels, 32, 14, 123)
        )

    def test_summed_kernels(self):
        tk = TK(
            word_embeddings=self.word_embedder,
            n_kernels=self.n_kernels,
            n_layers=self.n_layers,
            n_tf_dim=self.n_tf_dim,
            n_tf_heads=self.n_tf_heads,
        )
        query_embeddings = tk.word_embeddings(self.train_batch["query_tokens"])
        document_embeddings = tk.word_embeddings(
            self.train_batch["doc_pos_tokens"])
        contextualized_query_embeddings = tk.positional_encoding(
            query_embeddings)
        contextualized_document_embeddings = tk.positional_encoding(
            document_embeddings)
        transformed_query_embeddings = tk.transformer(
            contextualized_query_embeddings)
        transformed_document_embeddings = tk.transformer(
            contextualized_document_embeddings)
        translation_matrix = tk.create_translation_matrix(
            transformed_query_embeddings, transformed_document_embeddings)
        kernel_matrix = tk.apply_kernel_functions(translation_matrix)
        masked_kernel_matrix = tk.apply_masking(
            kernel_matrix, self.train_batch["query_tokens"], self.train_batch["doc_pos_tokens"])

        self.assertEquals(
            tk.apply_sums(masked_kernel_matrix).shape,
            (self.n_kernels, 32)
        )

    def test_fully_connected(self):
        tk = TK(
            word_embeddings=self.word_embedder,
            n_kernels=self.n_kernels,
            n_layers=self.n_layers,
            n_tf_dim=self.n_tf_dim,
            n_tf_heads=self.n_tf_heads,
        )
        query_embeddings = tk.word_embeddings(self.train_batch["query_tokens"])
        document_embeddings = tk.word_embeddings(
            self.train_batch["doc_pos_tokens"])
        contextualized_query_embeddings = tk.positional_encoding(
            query_embeddings)
        contextualized_document_embeddings = tk.positional_encoding(
            document_embeddings)
        transformed_query_embeddings = tk.transformer(
            contextualized_query_embeddings)
        transformed_document_embeddings = tk.transformer(
            contextualized_document_embeddings)
        translation_matrix = tk.create_translation_matrix(
            transformed_query_embeddings, transformed_document_embeddings)
        kernel_matrix = tk.apply_kernel_functions(translation_matrix)
        masked_kernel_matrix = tk.apply_masking(
            kernel_matrix, self.train_batch["query_tokens"], self.train_batch["doc_pos_tokens"])
        summed_kernels = tk.apply_sums(masked_kernel_matrix)

        self.assertEquals(
            tk.fully_connected(summed_kernels.T).shape,
            (32, 1)
        )
