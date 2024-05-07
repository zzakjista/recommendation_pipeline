import torch

class ModelConfig:
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
    def __init__(self):
        self.game_name:str = 'steam_games' # ['steam_games', 'amazon_games']
        self.bucket = None
        self.region = None

