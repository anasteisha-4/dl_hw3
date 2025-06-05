import numpy as np

def recall_at_k(targets, predict, k):
    hit_count = 0
    for true_id, pred_list in zip(targets, predict):
        top_k = pred_list[:k]
        if true_id in top_k:
            hit_count += 1
    return hit_count / len(targets)

def mrr_score(targets, predict):
    reciprocal_sum = 0.0
    for true_id, pred_list in zip(targets, predict):
        if true_id in pred_list:
            rank = pred_list.index(true_id) + 1
            reciprocal_sum += 1.0 / rank
    return reciprocal_sum / len(targets)