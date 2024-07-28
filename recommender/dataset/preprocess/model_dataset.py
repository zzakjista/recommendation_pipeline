import pandas as pd 
import numpy as np
import torch
import json
from collections import defaultdict
import scipy.sparse as sp
from .base import BaseDataset

class AEDataset(BaseDataset):
    """
    Autoencoder 모델 학습에 사용할 데이터셋을 로드하고, 각 모델에 필요한 데이터셋을 처리하는 기능을 제공하는 클래스
    """
    def __init__(self, cfg):
        super().__init__() # 부모 클래스의 생성자를 호출하여 상속받은 멤버 변수 초기화
        self.data_code = cfg.data.data_code
        self.user_item_dict = None
        self.train_data = {}
        self.valid_data = {}

        self.dataset = self.load(f'recommender/data/{self.data_code}_interaction_data.pkl')
        
        self.data = self.dataset['interaction']
        
        self.user2idx = self.encode(self.data['user_id'])
        self.idx2user = self.decode(self.data['user_id'])
        self.item2idx = self.encode(self.data['item_id'])
        self.idx2item = self.decode(self.data['item_id'])
        self.data['user_idx'] = self.data['user_id'].map(self.user2idx)
        self.data['item_idx'] = self.data['item_id'].map(self.item2idx)

    
        self.num_users = self.data['user_id'].nunique()
        self.num_items = self.data['item_id'].nunique()
        self.users = [i for i in range(self.num_users)]
        
        cfg.dataset.num_users = self.num_users
        cfg.dataset.num_items = self.num_items
        print(f'num_users: {cfg.dataset.num_users}, num_items: {cfg.dataset.num_items}')
        self.user_item_dict = self.make_user_item_dict()
        self.train_valid_split()

        with open(f'recommender/checkpoint/vocab.json', 'w') as f:
            json.dump(self.idx2item, f)


    def encode(self, feature) -> dict:
        return {v:i for i, v in enumerate(feature.unique())}
    
    def decode(self, feature) -> dict:
        return {str(i):str(v) for i, v in enumerate(feature.unique())}

    def get_data(self):
        return self.data
    
    def get_feature_names(self):
        return 
    
    def load(self, path:str):
        dataset = pd.read_pickle(path)
        return dataset
    
    def make_user_item_dict(self) -> dict:
        user_item_dict = defaultdict(list)
        for user, item in zip(self.data['user_idx'], self.data['item_idx']):
            user_item_dict[user].append(item)
        return user_item_dict
    
    def train_valid_split(self, valid_sample=1):
        assert self.user_item_dict is not None, 'user_item_dict is None. Run make_user_item_dict() first.'
        for user in self.user_item_dict:
            total = self.user_item_dict[user]
            valid = np.random.choice(self.user_item_dict[user], valid_sample, replace=False)
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
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset = self.load(f'recommender/data/{cfg.data.data_code}_interaction_data.pkl')
        
        self.data = self.dataset['interaction']
        self.user2idx = self.encode(self.data['user_id'])
        self.idx2user = self.decode(self.data['user_id'])
        self.item2idx = self.encode(self.data['item_id'])
        self.idx2item = self.decode(self.data['item_id'])
        self.data['user_idx'] = self.data['user_id'].map(self.user2idx)
        self.data['item_idx'] = self.data['item_id'].map(self.item2idx)

        self.num_users = self.data['user_id'].nunique()
        self.num_items = self.data['item_id'].nunique()
        cfg.dataset.num_users = self.num_users
        cfg.dataset.num_items = self.num_items
        self.users = [i for i in range(self.num_users)]
        print(f'num_users: {cfg.dataset.num_users}, num_items: {cfg.dataset.num_items}')
        
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
            valid = np.random.choice(self.user_item_dict[user], valid_sample, replace=False)
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
        mat = torch.zeros(size = (user_list.size(0), self.num_items))
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
    
