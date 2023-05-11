from core_metrics.core_metrics import calculate_metrics_plain, load_qrels
from model_tk.model_tk import *
from model_knrm.model_knrm import *
from data_loading import *
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.data.vocabulary import Vocabulary
import torch
from typing import Any, Callable, Tuple
from allennlp.common import Params, Tqdm
from allennlp.common.util import prepare_environment
from allennlp.data.dataloader import PyTorchDataLoader
prepare_environment(Params({}))  # sets the seeds to be fixed


print(f"Starting training script...")

# change paths to your data directory
config = {
    "vocab_directory": "../data/Part-2/allen_vocab_lower_10",
    "pre_trained_embedding": "../data/Part-2/glove.42B.300d.txt",
    "model": "knrm",
    "train_data": "../data/Part-2/triples.train.tsv",
    "validation_data": "../data/Part-2/tuples.validation.tsv",
    "test_data": "../data/Part-2/tuples.test.tsv",
    "qurels": "../data/Part-2/msmarco_qrels.txt"
}

# data loading
vocab = Vocabulary.from_files(config["vocab_directory"])
tokens_embedder = Embedding(vocab=vocab,
                            pretrained_file=config["pre_trained_embedding"],
                            embedding_dim=300,
                            trainable=True,
                            padding_index=0)
word_embedder = BasicTextFieldEmbedder({"tokens": tokens_embedder})


def get_model(model_name: str) -> torch.nn.Module:
    if model_name == "knrm":
        return KNRM(word_embedder, n_kernels=11)
    elif model_name == "tk":
        return TK(word_embedder, n_kernels=11,
                  n_layers=2, n_tf_dim=300, n_tf_heads=10)


model = get_model(config["model"])

criterion = torch.nn.HingeEmbeddingLoss()
optimizer = torch.optim.Adam(model.parameters())

print('Model', config["model"], 'total parameters:', sum(
    p.numel() for p in model.parameters() if p.requires_grad))
print('Network:', model)


def create_loader(data_path: str, create_loader: Callable[[], Any]) -> PyTorchDataLoader:
    _triple_reader = create_loader()
    _triple_reader = _triple_reader.read(data_path)
    _triple_reader.index_with(vocab)
    return PyTorchDataLoader(_triple_reader, batch_size=32)


loader = create_loader(
    config["train_data"],
    lambda: IrTripleDatasetReader(lazy=True, max_doc_length=180, max_query_length=30))

validation_loader = create_loader(
    config["validation_data"],
    lambda: IrLabeledTupleDatasetReader(lazy=True, max_doc_length=180, max_query_length=30))

qrel_dict = load_qrels(config["qurels"])
label = torch.ones(loader.batch_size)


def train_batch(batch: Dict):
    query = batch["query_tokens"]
    doc_pos = batch["doc_pos_tokens"]
    doc_neg = batch["doc_neg_tokens"]

    # Zero your gradients for every batch!
    optimizer.zero_grad()

    # Make predictions for this batch
    out_pos = model.forward(query, doc_pos)
    out_neg = model.forward(query, doc_neg)

    # Compute the loss and its gradients
    loss = criterion(out_pos, out_neg, label)
    loss.backward()

    # Adjust learning weights
    optimizer.step()


def validation_batch(batch: Dict) -> List[Tuple[Any, Any, float]]:
    query = batch["query_tokens"]
    doc = batch["doc_tokens"]

    out = model.forward(query, doc)

    return list(zip(batch["query_id"], batch["doc_id"], out.float().tolist()))


# train
metrics = []
best_mrr_at_10 = -1

for epoch in range(10):
    for batch in Tqdm.tqdm(loader):
        train_batch(batch)

    unsorted_validations = [validation_batch(
        batch) for batch in Tqdm.tqdm(validation_loader)]
    sorted_output = sorted(unsorted_validations, key=lambda x: x[2])
    validations_dict = {}
    for query_id, doc_id, ranking in sorted_output:
        if query_id not in validations_dict:
            validations_dict[query_id] = list()
        validations_dict[query_id].append(doc_id)

    metrics.append(calculate_metrics_plain(validations_dict, qrels))

    is_best_model_yet = metrics[-1]['MRR@10'] > best_mrr_at_10
    if is_best_model_yet:
        best_mrr_at_10 = metrics[-1]['MRR@10']
        torch.save(model.state_dict(), f'./models/{config["model"]}.pt')

    if epoch > 2:
        no_improvement_since_last_epoch = metrics[-1]['MRR@10'] < metrics[-2]['MRR@10']
        if no_improvement_since_last_epoch:
            break


# eval (duplicate for validation inside train loop - but rename "loader", since
# otherwise it will overwrite the original train iterator, which is instantiated outside the loop)
#

_tuple_reader = IrLabeledTupleDatasetReader(
    lazy=True, max_doc_length=180, max_query_length=30)
_tuple_reader = _tuple_reader.read(config["test_data"])
_tuple_reader.index_with(vocab)
loader = PyTorchDataLoader(_tuple_reader, batch_size=128)

for batch in Tqdm.tqdm(loader):
    # todo test loop
    # todo evaluation
    pass
