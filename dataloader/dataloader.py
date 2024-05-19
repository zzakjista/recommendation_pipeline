import torch
import torch.utils.data as data_utils
from abc import ABC, abstractmethod
class BaseDataLoader(ABC):

    @abstractmethod
    def __init__(self, dataset, args):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

class PytorchDataLoader(BaseDataLoader):
    """
    Pytorch로 학습하는 네트워크를 위한 DataLoader
    """
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.batch_size = args.batch_size
        # self.shuffle = args.shuffle
        # self.num_workers = args.num_workers
        self.data_loader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)

    def __iter__(self):
        return iter(self.data_loader)

    def __len__(self):
        return len(self.data_loader)

class MatrixDataLoader(BaseDataLoader):
    """
    Matrix 형태의 데이터를 처리하는 DataLoader
    Dataloader로서 기능은 없고, Matrix 데이터를 처리하기 위한 인터페이스를 제공한다.
    """

    def __init__(self, dataset, args):
        self.dataset = dataset

    def __iter__(self):
        pass

    def __len__(self):
        pass