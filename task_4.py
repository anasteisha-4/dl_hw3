from task_1 import recall_at_k, mrr_score
import torch
import gc
import os
from datasets import load_dataset
import numpy as np
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import random

def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def load_and_truncate_data(max_length=128):
    data = load_dataset("sentence-transformers/natural-questions", split="train")
    def truncate(example):
        return {
            "query": " ".join(example["query"].split()[:max_length]),
            "answer": " ".join(example["answer"].split()[:max_length])
        }
    return data.map(truncate, batched=False)

def train_model(train_samples, loss_type, model_name="intfloat/multilingual-e5-small", epochs=1):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        model_name,
        device=device,
        truncate_dim=128
    )
    loader = DataLoader(
        train_samples,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    if loss_type == 'contrastive':
        loss = losses.ContrastiveLoss(model=model)
    elif loss_type == 'triplet':
        loss = losses.TripletLoss(model=model)

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        optimizer_params={"lr": 1e-5},
        warmup_steps=50,
        output_path=f"e5-{loss_type}",
        show_progress_bar=True,
        checkpoint_save_steps=500,
        checkpoint_path="checkpoints"
    )
    return model

def evaluate_model(model, test_data):
    queries = ["query: " + q for q in test_data['query']]
    docs = ["passage: " + d for d in test_data['answer']]
    batch_encode = 64
    query_embs = model.encode(queries, batch_size=batch_encode, show_progress_bar=False)
    doc_embs = model.encode(docs, batch_size=batch_encode, show_progress_bar=False)
    
    sim_matrix = cosine_similarity(query_embs, doc_embs)
    all_predictions = np.argsort(-sim_matrix, axis=1)[:, :10].tolist()
    
    targets = list(range(len(queries)))
    return {
        "Recall@1":  recall_at_k(targets, all_predictions, 1),
        "Recall@3":  recall_at_k(targets, all_predictions, 3),
        "Recall@10": recall_at_k(targets, all_predictions, 10),
        "MRR": mrr_score(targets, all_predictions)
    }

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
torch.backends.cudnn.benchmark = True

clear_gpu_memory()
data = load_and_truncate_data()
train_data, test_data = data.train_test_split(test_size=0.2, seed=42).values()
indices = list(range(len(train_data)))
contrastive_samples = []
triplet_samples = []

for idx in indices:
    query = train_data[idx]["query"]
    pos_doc = train_data[idx]["answer"]
    
    neg_idx = random.choice(indices)
    while neg_idx == idx:
        neg_idx = random.choice(indices)
    neg_doc = train_data[neg_idx]["answer"]
    
    contrastive_samples.append(InputExample(
        texts=[f"query: {query}", f"passage: {pos_doc}"], 
        label=1.0
    ))
    contrastive_samples.append(InputExample(
        texts=[f"query: {query}", f"passage: {neg_doc}"], 
        label=0.0
    ))
    triplet_samples.append(InputExample(
        texts=[f"query: {query}", f"passage: {pos_doc}", f"passage: {neg_doc}"]
    ))

del data, train_data
clear_gpu_memory()

print("Training Contrastive Model")
model_contrastive = train_model(contrastive_samples, 'contrastive', epochs=1)
print("Evaluating Contrastive Model")
metrics_contrastive = evaluate_model(model_contrastive, test_data)
print("Contrastive Model Metrics:")
for k, v in metrics_contrastive.items():
    print(f"{k}: {v:.4f}")
del model_contrastive
clear_gpu_memory()

print("Training Triplet Model")
model_triplet = train_model(triplet_samples, 'triplet', epochs=1)
print("Evaluating Triplet Model")
metrics_triplet = evaluate_model(model_triplet, test_data)
print("Triplet Model Metrics:")
for k, v in metrics_triplet.items():
    print(f"{k}: {v:.4f}")