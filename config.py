import torch

class ModelConfig:
    """
    Model의 파라미터를 관리하는 클래스
    추후 argparse로 관리할 수 있음
    """
    def __init__(self):
        self.model_name:str = 'AutoRec'
        self.version:str = 'v0.0'
        self.batch_size:int = 32
        self.epochs:int = 100
        self.lr:float = 0.005
        self.optimizer:str = 'adam'
        self.criterion:str = 'mse'
        self.hidden_units:int = 500
        self.lamda_regularizer:float = 0.001
        self.seed:int = 42  
        self.verbose:int = 1
        self.device:str = 'cuda' if torch.cuda.is_available() else 'cpu'


class ServiceConfig:
    """
    Service의 파라미터를 관리하는 클래스
    튜닝과 관련되어있지 않아 argparse로 관리할 필요가 없음
    """
    def __init__(self):
        self.game_name:str = 'steam_games' # ['steam_games', 'amazon_games']
        self.bucket = None
        self.region = None

