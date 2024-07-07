from .collect import data_source_factory
from .preprocess import preprocessor_factory
from .preprocess import model_dataset_factory

def dataset_factory(args):
    dataset = data_source_factory(args)
    preprocessor_factory(dataset, args)
    dataset = model_dataset_factory(args)
    return dataset

    