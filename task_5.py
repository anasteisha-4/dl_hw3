from task_1 import recall_at_k, mrr_score
from datasets import load_dataset
import numpy as np
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import faiss

data = load_dataset("sentence-transformers/natural-questions", split="train")
seed = 42
train, test = data.train_test_split(test_size=0.2, seed=seed).values()

model = SentenceTransformer("intfloat/multilingual-e5-base")
train_samples = list(zip(train['query'], train['answer']))

docs = ["passage: " + d for _, d in train]
doc_embeddings = model.encode(docs, convert_to_numpy=True)
doc_embeddings = doc_embeddings.astype(np.float32)
doc_embeddings = np.ascontiguousarray(doc_embeddings)
faiss.normalize_L2(doc_embeddings)

index = faiss.IndexFlatIP(doc_embeddings.shape[1])
index.add(doc_embeddings)

triplets = []
for idx, (query, pos_doc) in enumerate(train):
    query_emb = model.encode("query: " + query, convert_to_numpy=True)
    query_emb = query_emb.reshape(1, -1).astype(np.float32)
    query_emb = np.ascontiguousarray(query_emb)
    faiss.normalize_L2(query_emb)
    
    D, I = index.search(query_emb, k=2)
    candidates = [int(i) for i in I[0] if i != idx][:1]
    
    for neg_idx in candidates:
        neg_doc = train_samples[neg_idx][1]
        triplets.append(
            InputExample(texts=["query: " + query, "passage: " + pos_doc, "passage: " + neg_doc])
        )
        
loader = DataLoader(triplets, batch_size=32)
train_loss = losses.TripletLoss(
    model=model,
    distance_metric=losses.TripletDistanceMetric.COSINE,
    triplet_margin=0.3
)

model.fit(
    train_objectives=[(loader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path="e5-hard-negatives",
    show_progress_bar=True
)
test_queries = ["query: " + q for q in test['query']]
test_docs = ["passage: " + d for d in test['answer']]

query_embs = model.encode(test_queries, batch_size=128)
doc_embs = model.encode(test_docs, batch_size=128)

sim_matrix = cosine_similarity(query_embs, doc_embs)
predictions = np.argsort(-sim_matrix, axis=1)[:, :10].tolist()
targets = list(range(len(test_queries)))

metrics = {
    'recall@1': recall_at_k(targets, predictions, 1),
    'recall@3': recall_at_k(targets, predictions, 3),
    'recall@10': recall_at_k(targets, predictions, 10),
    'mrr': mrr_score(targets, predictions)
}

print(f"""Recall@1: {metrics['recall@1']:.4f}
Recall@3: {metrics['recall@3']:.4f}
Recall@10: {metrics['recall@10']:.4f}
MRR: {metrics['mrr']:.4f}""")