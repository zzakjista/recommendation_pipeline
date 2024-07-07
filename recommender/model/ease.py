import torch
import numpy as np

class EASE:
    def __init__(self):
        self.X = None
        self.weight = None
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        """
        Convert scipy sparse matrix to PyTorch sparse matrix

        Arguments:
        ----------
        X = Adjacency matrix, scipy sparse matrix
        """
        coo = X.tocoo().astype(np.float32) # tocoo : COOrdinate format으로 변환 / COOrdinate format : 희소행렬을 나타내기 위한 방법
        i = torch.LongTensor(np.mat([coo.row, coo.col])) # row index, col index
        v = torch.FloatTensor(coo.data)
        res = torch.sparse.FloatTensor(i, v, coo.shape).to('cuda')
        return res

    def fit(self, reg:list):
        '''
        reg : regularization parameter
        '''
        G = self.X.to_dense().t() @ self.X.to_dense() # X^T * X
        diagIndices = torch.eye(G.shape[0]) == 1 # 대각선 index
        G[diagIndices] += reg  # regularization

        P = G.inverse() # inverse matrix
        B = P / (-1 * P.diag()) # B = -P / P_ii
        B[diagIndices] = 0 # 대각선은 0으로 만들어줌 / 왜냐하면 자기 자신과의 similarity는 0이기 때문
        self.weight = B
        # self.X.to_dense() @ B

        return None  # X * B