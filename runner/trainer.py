import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from abc import ABC, abstractmethod

from .metric import get_ndcg, get_hit

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
    def __init__(self, model, dataloader, args):
        self.model = model
        self.dataloader = dataloader
        self.args = args 

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


class AERunner(BaseRunner):

    def __init__(self, model, dataloader, args):
        super().__init__(model, dataloader, args)
        self.model = model
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.lr = args.lr
        self.device = args.device
        self.model = model.to(self.device)
        self.topk = args.topk
        self.num_epochs = args.num_epochs
        self.optimizer = self._create_optimizer(args.optimizer)
        self.criterion = self._create_criterion(args.criterion)

    def train(self):

        for epoch in range(1, self.num_epochs + 1):
            tbar = tqdm(range(1))
            for _ in tbar:
                loss = self.train_one_epoch()
                NDCG, HIT = self.evaluate()
                tbar.set_description(f'Epoch: {epoch:3d}| Train loss: {loss:.5f}| NDCG: {NDCG:.5f}| HIT: {HIT:.5f}')

    def train_one_epoch(self):
        self.model.train()
        loss_val = 0
        for users in self.dataloader:
            user_list = users.tolist()
            mat = self.dataset.get_matrix(user_list, trainYn=True)
            mat = mat.to(self.device)
            recon_mat = self.model(mat)

            self.optimizer.zero_grad()
            loss = self.criterion(recon_mat, mat)
            loss_val += loss.item()
            loss.backward()
            self.optimizer.step()

        loss_val /= len(self.dataloader)
        return loss_val
    
    def evaluate(self):
        self.model.eval()
        NDCG = 0.0 
        HIT = 0.0 
        with torch.no_grad():
            for users in self.dataloader:
                user_list = users.tolist()
                mat = self.dataset.get_matrix(user_list, trainYn=False).to(self.device)
                recon_mat = self.model(mat)
                recon_mat = recon_mat.softmax(dim = 1)
                recon_mat[mat == 1] = -1.
                rec_list = recon_mat.argsort(dim = 1)

                for user, rec in zip(users, rec_list):
                    user = user.item()
                    uv = self.dataset.valid_data[user] 
                    up = rec[-self.topk:].cpu().numpy().tolist()
                    NDCG += get_ndcg(pred_list = up, true_list = uv)
                    HIT += get_hit(pred_list = up, true_list = uv)

        NDCG /= len(self.dataloader.dataset)
        HIT /= len(self.dataloader.dataset)
        return NDCG, HIT
    
    def inference(self, user_ids):
        self.model.eval()

        user_list = [user_ids]
        mat = self.dataset.get_matrix(user_list, trainYn=False).to(self.device)
        recon_mat = self.model(mat)
        recon_mat = recon_mat.softmax(dim = 1)
        recon_mat[mat == 1] = -1.
        rec_list = recon_mat.argsort(dim = 1)
        rec_list = rec_list[0].cpu().numpy().tolist()
        rec_list = rec_list[-self.topk:]
        return rec_list

    def _create_optimizer(self, optimizer):
        if optimizer == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError('Invalid optimizer')

    def _create_criterion(self, criterion):
        if criterion == 'mse':
            return nn.MSELoss()
        elif criterion == 'ce':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError('Invalid criterion')

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')
        return None

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        print(f'Model loaded from {path}')
        return None
    


class EASERunner(BaseRunner):

    def __init__(self, model, dataloader, args):
        super().__init__(model, dataloader, args)
        self.model = model
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.device = args.device
        self.topk = args.topk
        self.num_epochs = args.num_epochs
        self.reg = args.reg

    def train(self):
        X = self.dataset.make_sparse_matrix(trainYn=True)
        for reg in self.reg:
            self.train_one_epoch(X, reg)
            NDCG, HIT = self.evaluate()
            print(f'NDCG:{NDCG} / HIT: {HIT}')
        return None
        
    def train_one_epoch(self, X, reg):
        self.model.X = self.model._convert_sp_mat_to_sp_tensor(X)
        self.model.fit(reg)
        return None
    
    def evaluate(self):
        NDCG = 0.0 
        HIT = 0.0 
        pred = self.model.pred.cpu()
        X = self.dataset.make_sparse_matrix(trainYn=True).toarray()
        mat = torch.from_numpy(X)
        pred[mat == 1] = -1
        pred = pred.argsort(dim = 1)

        for user, rec1 in tqdm(enumerate(pred)):
            uv = self.dataset.valid_data[user]

            # ranking
            up = rec1[-5:].cpu().numpy().tolist()[::-1]

            NDCG += get_ndcg(pred_list = up, true_list = uv)
            HIT += get_hit(pred_list = up, true_list = uv)

        NDCG /= len(self.dataset.train_data)
        HIT /= len(self.dataset.train_data)

        return NDCG, HIT
    
    def inference(self):
        pred = self.model.pred.cpu() # matrix 불러오기
        X = self.dataset.make_sparse_matrix(trainYn=False).toarray()
        mat = torch.from_numpy(X)
        pred[mat == 1] = -1
        # 유저 id row만 가져오기
        pred = pred.argsort(dim = 1) # dimension out of range 이유 : 
        # 각 유저의 top k 추천 아이템을 뽑아서 리스트로 만들어주기
        rec_list = {}
        for user, rec1 in enumerate(pred):
            up = rec1[-5:].cpu().numpy().tolist()[::-1]
            rec_list[user] = up

        # rec_list를 json으로 저장
        import json
        with open('rec_list.json', 'w') as f:
            json.dump(rec_list, f)

        return rec_list

    def save(self, path):
        pass

    def load(self, path):
        pass