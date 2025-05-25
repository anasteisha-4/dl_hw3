from task_1 import recall_at_k, mrr_score
from datasets import load_dataset
import numpy as np
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import random

data = load_dataset("sentence-transformers/natural-questions", split="train")
seed = 42
train, test = data.train_test_split(test_size=0.2, seed=seed).values()

contrastive_samples = []
triplet_samples = []

all_docs = [d for _, d in train]
query_to_doc = {q: d for q, d in train}

for query, pos_doc in train:
    candidate_docs = all_docs.copy()
    if query in query_to_doc:
        candidate_docs.remove(query_to_doc[query])
    neg_doc = random.choice(candidate_docs)
    contrastive_samples.append(InputExample(texts=[query, pos_doc], label=1.0))
    contrastive_samples.append(InputExample(texts=[query, neg_doc], label=0.0))
    
    triplet_samples.append(InputExample(texts=[query, pos_doc, neg_doc]))
    
def train_model(train_samples, loss_type, epochs=3):
    model = SentenceTransformer("intfloat/multilingual-e5-base")
    
    if loss_type == 'contrastive':
        loader = DataLoader(train_samples, batch_size=32)
        loss = losses.ContrastiveLoss(model=model)
    elif loss_type == 'triplet':
        loader = DataLoader(train_samples, batch_size=32)
        loss = losses.TripletLoss(model=model, distance_metric=losses.TripletDistanceMetric.COSINE)
    
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=100,
        output_path=f"e5-{loss_type}",
        show_progress_bar=True
    )
    return model

def evaluate_model(model, test_data):
    test_queries = ["query: " + q for q in test_data['query']]
    test_docs = ["passage: " + d for d in test_data['answer']]
    
    query_embs = model.encode(test_queries, batch_size=128)
    doc_embs = model.encode(test_docs, batch_size=128)
    
    sim_matrix = cosine_similarity(query_embs, doc_embs)
    predictions = np.argsort(-sim_matrix, axis=1)[:, :10].tolist()
    targets = list(range(len(test_queries)))
    return {
        'Recall@1': recall_at_k(targets, predictions, 1),
        'Recall@3': recall_at_k(targets, predictions, 3),
        'Recall@10': recall_at_k(targets, predictions, 10),
        'MRR': mrr_score(targets, predictions)
    }

print("Training Contrastive Model")
model_contrastive = train_model(contrastive_samples, 'contrastive')

print("Training Triplet Model")
model_triplet = train_model(triplet_samples, 'triplet')

metrics_contrastive = evaluate_model(model_contrastive, test)
metrics_triplet = evaluate_model(model_triplet, test)

print("Contrastive Model Metrics:")
for k, v in metrics_contrastive.items():
    print(f"{k}: {v:.4f}")

print("Triplet Model Metrics:")
for k, v in metrics_triplet.items():
    print(f"{k}: {v:.4f}")