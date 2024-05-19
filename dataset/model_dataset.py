import pandas as pd 
import numpy as np
import torch
from collections import defaultdict
import scipy.sparse as sp

from .base import BaseDataset

class AEDataset(BaseDataset):
    """
    Autoencoder 모델 학습에 사용할 데이터셋을 로드하고, 각 모델에 필요한 데이터셋을 처리하는 기능을 제공하는 클래스
    """
    def __init__(self, args):
        super().__init__() # 부모 클래스의 생성자를 호출하여 상속받은 멤버 변수 초기화
        self.game_name = args.game_name
        self.user_item_dict = None
        self.train_data = {}
        self.valid_data = {}

        self.data = self.load(f'data/{self.game_name}_interaction_data.pkl')
        self.data['user_idx'] = self.data['user_id'].map(self.encode(self.data['user_id']))
        self.data['item_idx'] = self.data['item_id'].map(self.encode(self.data['item_id']))

        self.users = [i for i in range(self.num_users)]

        self.num_users = self.data['user_id'].nunique()
        self.num_items = self.data['item_id'].nunique()
        args.num_users = self.num_users
        args.num_items = self.num_items

        self.user_item_dict = self.make_user_item_dict()

    def encode(self, feature) -> dict:
        return {v:i for i, v in enumerate(feature.unique())}
    
    def decode(self, feature) -> dict:
        return {i:v for i, v in enumerate(feature.unique())}

    def get_data(self):
        return self.data
    
    def get_feature_names(self):
        return 
    
    def load(self, path:str):
        self.data = pd.read_pickle(path)
        self.num_users = self.data['user_id'].nunique()
        self.num_items = self.data['item_id'].nunique()
        return self.data
    
    def make_user_item_dict(self) -> dict:
        user_item_dict = defaultdict(list)
        for user, item in zip(self.data['user_idx'], self.data['item_idx']):
            user_item_dict[user].append(item)
        return user_item_dict
    
    def train_valid_split(self, valid_sample=5):
        assert self.user_item_dict is not None, 'user_item_dict is None. Run make_user_item_dict() first.'
        for user in self.user_item_dict:
            total = self.user_item_dict[user]
            valid = np.random.choice(self.user_item_dict[user], valid_sample, replace=True)
            train = np.setdiff1d(total, valid)
            self.train_data[user] = list(train)
            self.valid_data[user] = list(valid)
        return None
    
    def get_matrix(self, user_list:list, trainYn=True) -> torch.FloatTensor:
        """
        AutoEncoder 모델 학습에 사용할 데이터셋을 만드는 함수
        trainYn : True이면 train 데이터셋을, False이면 valid 데이터셋을 만든다.
        """
        mat = torch.zeros((len(user_list), self.num_items))
        for idx, user in enumerate(user_list):
            if trainYn:
                mat[idx, self.train_data[user]] = 1
            else:
                mat[idx, self.user_item_dict[user]] = 1
        return torch.FloatTensor(mat)
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return self.users[idx]
    

class EASEDataset(BaseDataset):
    """
    EASE 모델 학습에 사용할 데이터셋을 로드하고, 각 모델에 필요한 데이터셋을 처리하는 기능을 제공하는 클래스
    """
    def __init__(self, args):
        self.args = args
        self.data = self.load(f'data/{args.game_name}_interaction_data.pkl')
        self.data['user_idx'] = self.data['user_id'].map(self.encode(self.data['user_id']))
        self.data['item_idx'] = self.data['item_id'].map(self.encode(self.data['item_id']))

        self.num_users = self.data['user_id'].nunique()
        self.num_items = self.data['item_id'].nunique()
        args.num_users = self.num_users
        args.num_items = self.num_items
        self.users = [i for i in range(self.num_users)]
        
        self.train_data = {}
        self.valid_data = {}
        self.user_item_dict = self.make_user_item_dict()

    def encode(self, feature) -> dict:
        return {v:i for i, v in enumerate(feature.unique())}
    
    def decode(self, feature) -> dict:
        return {i:v for i, v in enumerate(feature.unique())}

    def get_data(self):
        return self.data
    
    def get_feature_names(self):
        return super().get_feature_names()
    
    def load(self, path):
        data = pd.read_pickle(path)
        return data
    
    def make_user_item_dict(self) -> dict:
        user_item_dict = defaultdict(list)
        for user, item in zip(self.data['user_idx'], self.data['item_idx']):
            user_item_dict[user].append(item)
        return user_item_dict
    
    def train_valid_split(self, valid_sample=5):
        assert self.user_item_dict is not None, 'user_item_dict is None. Run make_user_item_dict() first.'
        for user in self.user_item_dict:
            total = self.user_item_dict[user]
            valid = np.random.choice(self.user_item_dict[user], valid_sample, replace=True)
            train = np.setdiff1d(total, valid)
            self.train_data[user] = list(train)
            self.valid_data[user] = list(valid)
        return None
    
    def get_train_valid_data(self):
        return self.user_train, self.user_valid

    def make_matrix(self, user_list, trainYn=True):
        """
        user_item_dict를 바탕으로 행렬 생성
        """
        mat = torch.zeros(size = (user_list.size(0), self.num_item))
        for idx, user in enumerate(user_list):
            if trainYn:
                mat[idx, self.train_data[user.item()]] = 1
            else:
                mat[idx, self.user_item_dict[user.item()]] = 1
        return mat

    def make_sparse_matrix(self, trainYn=True):
        X = sp.dok_matrix((self.num_users, self.num_items), dtype=np.float32)
        if trainYn:
            for user in self.train_data.keys():
                item_list = self.train_data[user]
                X[user, item_list] = 1.0
        else:
            for user in self.user_item_dict.keys():
                item_list = self.user_item_dict[user]
                X[user, item_list] = 1.0
        return X.tocsr()
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return self.users[idx]
