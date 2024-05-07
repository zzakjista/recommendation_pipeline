from .dataloader import PytorchDataLoader

def dataloader_factory(dataset, batch_size:int, shuffle:bool, num_workers:int):
    return PytorchDataLoader(dataset, batch_size, shuffle, num_workers)
