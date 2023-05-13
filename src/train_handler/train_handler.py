import torch
from ..config.config import Config
from ..core_metrics.core_metrics import calculate_metrics_plain, load_qrels
from ..model_tk.model_tk import TK
from ..model_knrm.model_knrm import KNRM
from ..data_loading import IrTripleDatasetReader, IrLabeledTupleDatasetReader
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import move_to_device
import torch
import torch.nn
from typing import Any, Callable, Tuple, Dict, List
from allennlp.common import Params, Tqdm
from allennlp.common.util import prepare_environment
from allennlp.data.dataloader import PyTorchDataLoader

BATCH_SIZE = 32


def get_model(model_name: str, word_embedder: TextFieldEmbedder) -> torch.nn.Module:
    if model_name == "knrm":
        return KNRM(word_embedder, n_kernels=11)
    elif model_name == "tk":
        return TK(word_embedder, n_kernels=11,
                  n_layers=2, n_tf_dim=300, n_tf_heads=10)


def create_loader(data_path: str, create_loader: Callable[[], Any], vocab: Vocabulary) -> PyTorchDataLoader:
    _triple_reader = create_loader()
    _triple_reader = _triple_reader.read(data_path)
    _triple_reader.index_with(vocab)
    # TODO refactor TQDM into own wrapper
    return Tqdm.tqdm(PyTorchDataLoader(_triple_reader, batch_size=BATCH_SIZE))


class PlainMetricsCalculator:
    def calculate_metrics(self, ranking, qrels, binarization_point=1.0, return_per_query=False):
        return calculate_metrics_plain(ranking, qrels, binarization_point, return_per_query)


class TrainHandler():
    @classmethod
    def from_config(cls, config: Config):
        vocab = Vocabulary.from_files(config.vocab_directory)
        tokens_embedder = Embedding(vocab=vocab,
                                    pretrained_file=config.pre_trained_embedding,
                                    embedding_dim=300,
                                    trainable=True,
                                    padding_index=0)
        word_embedder = BasicTextFieldEmbedder({"tokens": tokens_embedder})
        qrels = load_qrels(config.qurels)
        model = get_model(config.model, word_embedder)
        qrels = load_qrels(config.qurels)
        criterion = torch.nn.MarginRankingLoss(margin=1, reduction='mean')
        optimizer = torch.optim.Adam(model.parameters())

        train_data_loader = create_loader(
            config.train_data,
            lambda: IrTripleDatasetReader(
                lazy=True, max_doc_length=180, max_query_length=30),
            vocab)

        validation_data_loader = create_loader(
            config.validation_data,
            lambda: IrLabeledTupleDatasetReader(
                lazy=True, max_doc_length=180, max_query_length=30),
            vocab)

        return cls(
            model_type=config.model,
            model=model,
            vocab=vocab,
            word_embedder=word_embedder,
            criterion=criterion,
            optimizer=optimizer,
            qrels=qrels,
            train_data_loader=train_data_loader,
            validation_data_loader=validation_data_loader,
            metric_calculator=PlainMetricsCalculator(),
            save_to_disk=True,
            save_models_dir=config.save_models_dir
        )

    def __init__(self,
                 model_type: str,
                 model: torch.nn.Module,
                 vocab: Vocabulary,
                 word_embedder: BasicTextFieldEmbedder,
                 criterion,
                 optimizer,
                 qrels,
                 train_data_loader,
                 validation_data_loader,
                 metric_calculator,
                 save_to_disk: bool,
                 save_models_dir: str):

        self.model_type = model_type
        self.model = model
        self.vocab = vocab
        self.word_embedder = word_embedder
        self.criterion = criterion
        self.optimizer = optimizer
        self.qrels = qrels
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.metric_calculator = metric_calculator
        self.save_to_disk = save_to_disk
        self.save_models_dir = save_models_dir

        self.device = self.get_device()

    def get_device(self) -> torch.device:
        if torch.cuda.is_available():
            print("CUDA available. GPU will be used for training training.")
            return torch.device("cuda:0")
        else:
            print("CUDA unavailable. CPU will be used for training.")
            return torch.device("cpu:0")

    def move_to_device(self, obj):
        if torch.cuda.is_available():
            return move_to_device(obj, self.device)
        else:
            return obj

    def train_batch(self, batch: Dict, true_labels: torch.Tensor):
        batch = self.move_to_device(batch)
        query = batch["query_tokens"]
        doc_pos = batch["doc_pos_tokens"]
        doc_neg = batch["doc_neg_tokens"]

        # Zero your gradients for every batch!
        self.optimizer.zero_grad()

        # Make predictions for this batch
        out_pos = self.model.forward(query, doc_pos)
        out_neg = self.model.forward(query, doc_neg)

        # Compute the loss and its gradients
        loss = self.criterion(out_pos, out_neg, true_labels)
        loss.backward()

        # Adjust learning weights
        self.optimizer.step()

    def validation_batch(self, batch: Dict) -> List[Tuple[Any, Any, float]]:
        batch = self.move_to_device(batch)
        query = batch["query_tokens"]
        doc = batch["doc_tokens"]

        out = self.model.forward(query, doc)

        return list(zip(batch["query_id"], batch["doc_id"], out.float().tolist()))

    def train(self):
        print(f"Start training...")

        prepare_environment(Params({}))  # sets the seeds to be fixed

        if torch.cuda.is_available():
            self.model.cuda()
            self.criterion.cuda()

        print('Model', self.model_type, 'total parameters:', sum(
            p.numel() for p in self.model.parameters() if p.requires_grad))
        print('Network:', self.model)

        true_labels = torch.ones(BATCH_SIZE).to(self.device)

        # train
        metrics = []
        best_mrr_at_10 = -1

        for epoch in range(10):
            for batch in self.train_data_loader:
                self.train_batch(batch, true_labels)

            unsorted_validations = []
            for batch in self.validation_data_loader:
                unsorted_validations.extend(self.validation_batch(batch))
            sorted_output = sorted(unsorted_validations, key=lambda x: x[2])
            validations_dict = {}
            for query_id, doc_id, ranking in sorted_output:
                if query_id not in validations_dict:
                    validations_dict[query_id] = list()
                validations_dict[query_id].append(doc_id)

            metrics.append(self.metric_calculator.calculate_metrics(
                validations_dict, self.qrels))

            is_best_model_yet = metrics[-1]['MRR@10'] > best_mrr_at_10
            if is_best_model_yet:
                best_mrr_at_10 = metrics[-1]['MRR@10']
                if self.save_to_disk:
                    model_path = f'{self.save_models_dir}/{self.model_type}-epoch-{epoch}.pt'
                    torch.save(self.model.state_dict(), model_path)
                    print(
                        f"New best model: MRR@10 = {best_mrr_at_10}; saved to \"{model_path}\"")

            if epoch > 2:
                no_improvement_since_last_epoch = metrics[-1]['MRR@10'] < metrics[-2]['MRR@10']
                if no_improvement_since_last_epoch:
                    break
