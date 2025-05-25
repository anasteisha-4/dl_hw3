from task_1 import recall_at_k, mrr_score
from datasets import load_dataset
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

data = load_dataset("sentence-transformers/natural-questions", split="train")
seed = 42
train, test = data.train_test_split(test_size=0.2, seed=seed).values()

model = SentenceTransformer("intfloat/multilingual-e5-base", device="cuda")
queries = ["query: " + q for q in test['query']]
passages = ["passage: " + a for a in test['answer']]

batch_size = 128
top_k = 10

def encode_in_batches(texts, batch_size=128):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, convert_to_numpy=True)
        embeddings.append(emb)
    return np.vstack(embeddings)

query_embeddings = encode_in_batches(queries)
passage_embeddings = encode_in_batches(passages)

sim_matrix = cosine_similarity(query_embeddings, passage_embeddings)
predictions = np.argsort(-sim_matrix, axis=1)[:, :top_k].tolist()
targets = list(range(len(test['query'])))
metrics = {
    'recall@1': recall_at_k(targets, predictions, 1),
    'recall@3': recall_at_k(targets, predictions, 3),
    'recall@10': recall_at_k(targets, predictions, 10),
    'mrr': mrr_score(targets, predictions)
}

print(f"""Recall@1: {metrics['recall@1']:.6f}
Recall@3: {metrics['recall@3']:.6f}
Recall@10: {metrics['recall@10']:.6f}
MRR: {metrics['mrr']:.6f}""")