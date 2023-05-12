# eval (duplicate for validation inside train loop - but rename "loader", since
# otherwise it will overwrite the original train iterator, which is instantiated outside the loop)
#

_tuple_reader = IrLabeledTupleDatasetReader(
    lazy=True, max_doc_length=180, max_query_length=30)
_tuple_reader = _tuple_reader.read(config.test_data)
_tuple_reader.index_with(vocab)
loader = PyTorchDataLoader(_tuple_reader, batch_size=128)

for batch in Tqdm.tqdm(loader):
    # todo test loop
    # todo evaluation
    pass
