from abc import ABC, abstractmethod

class BaseDataset(ABC):
    """
    모델 학습에 사용할 데이터셋을 로드하고, 각 모델에 필요한 데이터셋을 처리하는 기능을 제공하는 추상 클래스
    """
    def __init__(self):
        self.data = None
        self.num_users = 0
        self.num_items = 0
        self.feature_names = None

    @abstractmethod
    def encode(self):
        pass

    @abstractmethod
    def decode(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def get_feature_names(self):
        pass

    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass