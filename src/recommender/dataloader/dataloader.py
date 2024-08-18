import torch.utils.data as data_utils
from abc import ABC, abstractmethod
class BaseDataLoader(ABC):

    @abstractmethod
    def __init__(self, dataset, cfg):
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
    def __init__(self, dataset, cfg):
        self.dataset = dataset
        self.batch_size = cfg.dataloader.batch_size
        self.dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

class MatrixDataLoader(BaseDataLoader):
    """
    Matrix 형태의 데이터를 처리하는 DataLoader
    Dataloader로서 기능은 없고, Matrix 데이터를 처리하기 위한 인터페이스를 제공한다.
    """

    def __init__(self, dataset, cfg):
        self.dataset = dataset

    def __iter__(self):
        pass

    def __len__(self):
        pass