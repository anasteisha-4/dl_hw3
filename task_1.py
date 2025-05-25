import numpy as np

def recall_at_k(targets, predictions, k):
    hits = []
    for target, pred in zip(targets, predictions):
        hit = 1 if target in pred[:k] else 0
        hits.append(hit)
    return np.mean(hits)

def mrr_score(targets, predictions):
    reciprocal_ranks = []
    for target, pred in zip(targets, predictions):
        if target in pred:
            rank = pred.index(target) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)
    return np.mean(reciprocal_ranks)