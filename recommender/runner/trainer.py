import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from hydra.utils import instantiate
from tqdm import tqdm
from .metric import get_ndcg, get_hit
from .base import BaseRunner


class AERunner(BaseRunner):

    def __init__(self, model, dataloader, cfg):
        super().__init__(model, dataloader, cfg)
        self.model = model
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.lr = cfg.optimizer.lr
        self.device = cfg.device
        self.model = model.to(self.device)
        self.topk = cfg.topk
        self.num_epochs = cfg.runner.num_epochs
        self.optimizer = instantiate(cfg.optimizer, model.parameters(), lr=self.lr) # hydra util로 instance 생성
        self.criterion = instantiate(cfg.criterion)

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
                mat = self.dataset.get_matrix(user_list, trainYn=True).to(self.device)
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
    
    def inference(self, user_ids=None):
        user_ids = user_ids if user_ids else self.dataset.users[0]
        self.model.eval()
        with torch.no_grad():
            user_list = [user_ids]
            mat = self.dataset.get_matrix(user_list, trainYn=False).to(self.device)
            recon_mat = self.model(mat)
            recon_mat = recon_mat.softmax(dim = 1)
            recon_mat[mat == 1] = -1.
            rec_list = recon_mat.argsort(dim = 1)
            rec_list = rec_list[0].cpu().numpy().tolist()
            rec_list = rec_list[-self.topk:]
        rec_list = [self.dataset.idx2item[i] for i in rec_list]
        print(f'User {user_ids} : {rec_list}')
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
    
import random
class EASERunner(BaseRunner):

    def __init__(self, model, dataloader, cfg):
        super().__init__(model, dataloader, cfg)
        self.model = model
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.device = cfg.device
        self.topk = cfg.topk
        self.num_epochs = cfg.num_epochs
        self.reg = cfg.reg

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
    
    def inference(self, user_ids=None):
        user_ids = random.choice(list(self.dataset.user_item_dict.keys()))
        user_list = [user_ids]
        mat = self.dataset.get_matrix(user_list, trainYn=False)
        recon_mat = mat @ self.model.weight
        recon_mat = recon_mat.cpu()
        recon_mat[mat == 1] = -1
        rec_list = recon_mat.argsort(dim = 1)
        rec_list = rec_list[0].cpu().numpy().tolist()
        rec_list = rec_list[-self.topk:][::-1]
        return rec_list

    def batch_inference(self):
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