from .collect import data_source_factory
from .preprocess import preprocessor_factory
from .preprocess import model_dataset_factory

def dataset_factory(cfg):
    dataset = data_source_factory(cfg)
    preprocessor_factory(dataset, cfg)
    dataset = model_dataset_factory(cfg)
    return dataset

    