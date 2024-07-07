import torch
import pandas as pd
import numpy as np
import random

class AEGenerator:
    """
    Autoencoder의 실시간 추론을 구현한 Generator 클래스
    """

    def __init__(self, model, args, vocab):
        self.model = model
        self.num_items = args.num_items
        self.vocab = vocab

    def generate(self, user_id):
        """
        유저 정보를 받아 추천 결과를 생성하는 기능
        :input: 
            - user_id
        :return:
            - response
        *특이 사항
            - DB 혹은 저장소에서 유저의 interaction 정보를 받아오는 것은 구현되지 않음
        """
        user_id = [user_id]
        mat = self.get_matrix()
        recon_mat = self.model.run(mat)
        recon_mat = recon_mat.softmax(dim=1)
        recon_mat[mat == 1] = -1.
        rec_list = recon_mat.argsort(dim=1)
        rec_list = rec_list[0].cpu().numpy().tolist()
        rec_list = rec_list[-10:]
        rec_list = [self.vocab[str(i)] for i in rec_list]
        response = {'items': rec_list}
        return response

    def get_matrix(self, interaction=None):
        """
        유저의 interaction 정보를 1차원의 matrix 형태로 반환하는 함수
        :input: 
            - interaction
        :return:
            - torch.FloatTensor
        *특이 사항
            - 현재는 구현을 위해 random item을 생성함
        """
        mat = torch.zeros((1, self.num_items))
        random_item = [random.randint(0, self.num_items-1) for i in range(10)]
        mat[:, random_item] = 1
        return mat.to('cpu')