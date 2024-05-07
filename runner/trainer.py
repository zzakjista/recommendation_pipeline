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
    def __init__(self, model, dataloader, optimizer, criterion, lr, device, scheduler=None):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optim.Adam(model.parameters(), lr=lr) if optimizer=='adam' else optim.SGD(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss() if criterion=='mse' else nn.CrossEntropyLoss()
        self.device = device
        self.scheduler = scheduler

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

    def __init__(self, model, dataloader, optimizer, criterion, lr, device, dataset, scheduler=None):
        super().__init__(model, dataloader, optimizer, criterion, lr, device, scheduler)
        self.dataset = dataset
        self.model = model.to(device)

    def train(self, num_epochs):

        for epoch in range(1, num_epochs + 1):
            tbar = tqdm(range(1))
            for _ in tbar:
                loss = self.train_one_epoch()
                NDCG, HIT = self.evaluate()
                tbar.set_description(f'Epoch: {epoch:3d}| Train loss: {loss:.5f}| NDCG@10: {NDCG:.5f}| HIT@10: {HIT:.5f}')
            if self.scheduler:
                self.scheduler.step()

    def train_one_epoch(self):
        self.model.train()
        loss_val = 0
        for users in self.dataloader:
            user_list = users.tolist()
            mat = self.dataset.get_matrix(user_list)
            mat = mat.to(self.device)
            recon_mat = self.model(mat)

            self.optimizer.zero_grad()
            loss = self.criterion(recon_mat, mat)
            loss_val += loss.item()
            loss.backward()
            self.optimizer.step()

        loss_val /= len(self.dataloader)
        return loss_val
    
    def evaluate(self, top_k=5):
        self.model.eval()

        NDCG = 0.0 # NDCG@10
        HIT = 0.0 # HIT@10

        with torch.no_grad():
            for users in self.dataloader:
                user_list = users.tolist()
                mat = self.dataset.get_matrix(user_list).to(self.device)
                recon_mat = self.model(mat)
                recon_mat = recon_mat.softmax(dim = 1)
                recon_mat[mat == 1] = -1.
                rec_list = recon_mat.argsort(dim = 1)

                for user, rec in zip(users, rec_list):
                    user = user.item()
                    uv = self.dataset.valid_data[user] 
                    up = rec[-top_k:].cpu().numpy().tolist()
                    NDCG += get_ndcg(pred_list = up, true_list = uv)
                    HIT += get_hit(pred_list = up, true_list = uv)

        NDCG /= len(self.dataloader.dataset)
        HIT /= len(self.dataloader.dataset)
        return NDCG, HIT
    
    def inference(self, user_ids):
        pass

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')
        return None

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        print(f'Model loaded from {path}')
        return None