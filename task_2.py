from task_1 import recall_at_k, mrr_score
from datasets import load_dataset
import numpy as np
from collections import defaultdict
import re
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.metrics.pairwise import cosine_similarity
import gc
from sklearn.feature_extraction.text import TfidfVectorizer

data = load_dataset("sentence-transformers/natural-questions", split="train")
seed = 42
train, test = data.train_test_split(test_size=0.2, seed=seed).values()
train = train['query'] + train['answer']
vectorizer = TfidfVectorizer(
    max_features=200000,
    ngram_range=(1,3),
    analyzer='char_wb',
    min_df=0.0001, 
    max_df=0.9,
    sublinear_tf=False
)

vectorizer.fit(train)
X = vectorizer.transform(test['answer']).astype(np.float32)
doc_file = 'doc.npz'
save_npz(doc_file, X)
del X
gc.collect()

X_questions = vectorizer.transform(test['query']).astype(np.float32)
X_docs = load_npz(doc_file)

sim_matrix = cosine_similarity(X_questions, X_docs)
predictions = np.argsort(-sim_matrix, axis=1)[:, :10].tolist()
targets = list(range(len(test['query'])))

del X_questions, sim_matrix, X_docs
gc.collect()

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
