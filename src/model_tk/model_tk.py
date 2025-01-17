from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.autograd import Variable
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from .positional_encoding.positional_encoding import PositionalEncoding


class TK(nn.Module):
    '''
    Paper: S. Hofstätter, M. Zlabinger, and A. Hanbury 2020. Interpretable & Time-Budget-Constrained Contextualization for Re-Ranking. In Proc. of ECAI 
    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 n_kernels: int,
                 n_layers: int,
                 n_tf_dim: int,
                 n_tf_heads: int):

        super(TK, self).__init__()

        self.n_kernels = n_kernels
        self.n_layers = n_layers
        self.n_tf_dim = n_tf_dim
        self.n_tf_heads = n_tf_heads

        self.word_embeddings = word_embeddings

        # static - kernel size & magnitude variables
        mu = torch.FloatTensor(self.kernel_mus(
            n_kernels)).view(1, 1, 1, n_kernels)
        sigma = torch.FloatTensor(self.kernel_sigmas(
            n_kernels)).view(1, 1, 1, n_kernels)

        self.register_buffer('mu', mu)
        self.register_buffer('sigma', sigma)

        embedding_dimension = self.word_embeddings.get_output_dim()

        # setting up the contextualization using sinusoid curves
        self.positional_encoding = PositionalEncoding(embedding_dimension)

        # create transformer with attention
        encoder = nn.TransformerEncoderLayer(
            embedding_dimension, n_tf_heads, n_tf_dim)
        self.transformer = nn.TransformerEncoder(encoder, n_layers)

        self.fully_connected = nn.Linear(n_kernels, 1, bias=False)
        nn.init.xavier_uniform_(self.fully_connected.weight)

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        # shape: (batch, query_max,emb_dim)
        query_embeddings = self.word_embeddings(query)
        # shape: (batch, document_max,emb_dim)
        document_embeddings = self.word_embeddings(document)

        contextualized_query_embeddings = self.positional_encoding(
            query_embeddings)
        contextualized_document_embeddings = self.positional_encoding(
            document_embeddings)

        transformed_query_embeddings = self.transformer(
            contextualized_query_embeddings)
        transformed_document_embeddings = self.transformer(
            contextualized_document_embeddings)

        translation_matrix = self.create_translation_matrix(
            transformed_query_embeddings, transformed_document_embeddings)
        kernel_matrix = self.apply_kernel_functions(translation_matrix)
        masked_kernel_matrix = self.apply_masking(
            kernel_matrix, query, document)
        summed_kernels = self.apply_sums(masked_kernel_matrix)
        return self.fully_connected(summed_kernels.T).squeeze(1)

    def kernel_mus(self, n_kernels: int):
        """
        get the mu for each guassian kernel. Mu is the middle of each bin
        :param n_kernels: number of kernels (including exact match). first one is exact match
        :return: l_mu, a list of mu.
        """
        l_mu = [1.0]
        if n_kernels == 1:
            return l_mu

        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu

    def kernel_sigmas(self, n_kernels: int):
        """
        get sigmas for each guassian kernel.
        :param n_kernels: number of kernels (including exactmath.)
        :param lamb:
        :param use_exact:
        :return: l_sigma, a list of simga
        """
        bin_size = 2.0 / (n_kernels - 1)
        l_sigma = [0.0001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [0.5 * bin_size] * (n_kernels - 1)
        return l_sigma

    def create_translation_matrix(self, query: torch.Tensor, document: torch.Tensor) -> torch.Tensor:
        normed_queries = query / \
            torch.functional.norm(query, dim=-1, keepdim=True)
        normed_document = document / \
            torch.functional.norm(document, dim=-1, keepdim=True)

        translation_matrix = torch.bmm(
            normed_queries, normed_document.permute(0, 2, 1))

        return translation_matrix

    def apply_kernel_functions(self, translation_matrix: torch.Tensor) -> torch.Tensor:
        batch_size, query_size, doc_size = translation_matrix.shape
        K = torch.zeros(self.n_kernels, batch_size, query_size, doc_size)
        if torch.cuda.is_available():
            K = K.cuda()
        for k in range(self.n_kernels):
            K[k] = TK.gaussian(translation_matrix,
                                 self.mu[..., k], self.sigma[..., k])

        return K

    def apply_masking(self, kernel_matrix: torch.Tensor, query, document) -> torch.Tensor:
        # shape: (batch, query_max)
        # > 1 to also mask oov terms
        query_pad_oov_mask = (query["tokens"]["tokens"] > 0).float()
        # shape: (batch, doc_max)
        document_pad_oov_mask = (document["tokens"]["tokens"] > 0).float()

        _, query_shape = query_pad_oov_mask.shape
        _, doc_shape = document_pad_oov_mask.shape

        masked_kernels = torch.mul(
            kernel_matrix, query_pad_oov_mask.view(1, _, query_shape, 1))
        masked_kernels = torch.mul(
            masked_kernels, document_pad_oov_mask.view(1, _, 1, doc_shape))

        return masked_kernels

    def apply_sums(self, kernel_matrix: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.log(torch.sum(kernel_matrix, dim=-1)), dim=-1)

    @staticmethod
    def gaussian(x: torch.Tensor, mu: float, sigma: float):
        return torch.exp(
            - torch.pow(x - mu, 2) / (2*torch.pow(sigma, 2))
        )
