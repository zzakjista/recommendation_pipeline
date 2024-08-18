from abc import ABC, abstractmethod


class BaseRunner(ABC):
    """
    Abstract class for training and testing PyTorch models.
    하위 클래스에서 필수적으로 구현해야하는 메소드는 아래와 같습니다.
    - train : epochs만큼 모델을 학습할 수 있는 기능
    - train_one_epoch : 1 epoch만큼 모델을 학습할 수 있는 기능
    - evaluate : 모델 성능을 모니터링하는 기능
    - inference : 학습된 모델을 사용하여 추론을 수행하는 기능
    - save : 모델의 checkpoint를 저장하는 기능
    - load : 모델의 checkpoint를 불러오는 기능
    """
    def __init__(self, model, dataloader, cfg):
        self.model = model
        self.dataloader = dataloader
        self.cfg = cfg 

    @abstractmethod
    def train(self, train_loader, epoch):
        pass

    @abstractmethod
    def train_one_epoch(self, train_loader, epoch):
        pass

    @abstractmethod
    def evaluate(self, val_loader):
        pass

    @abstractmethod
    def inference(self, user_ids):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

