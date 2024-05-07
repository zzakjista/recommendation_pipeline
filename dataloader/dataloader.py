import torch
import torch.utils.data as data_utils
from abc import ABC, abstractmethod
class BaseDataLoader(ABC):

    @abstractmethod
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

class PytorchDataLoader(BaseDataLoader):

    def __init__(self, dataset, batch_size:int, shuffle:bool, num_workers:int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.data_loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def __iter__(self):
        return iter(self.data_loader)

    def __len__(self):
        return len(self.data_loader)
