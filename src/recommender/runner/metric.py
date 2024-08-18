import numpy as np

def get_ndcg(pred_list, true_list):
    ndcg = 0
    for rank, pred in enumerate(pred_list):
        if pred in true_list:
            ndcg += 1 / np.log2(rank + 2)
    return ndcg

# hit == recall == precision
def get_hit(pred_list, true_list):
    hit_list = set(true_list) & set(pred_list)
    hit = len(hit_list) / len(true_list)
    return hit