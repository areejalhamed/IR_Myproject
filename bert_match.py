#bert_match
import os
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

REPRESENTATIONS_DIR = "representations"

def match_query_bert(table_name):
    bert_file = os.path.join(REPRESENTATIONS_DIR, f"bert_embeddings_{table_name}.pkl")
    query_file = os.path.join(REPRESENTATIONS_DIR, "queries", f"bert_query_{table_name}.pkl")
    
    if not os.path.exists(bert_file) or not os.path.exists(query_file):
        raise FileNotFoundError("❌ لم يتم العثور على ملفات التمثيل المطلوبة.")
    
    data = joblib.load(bert_file)
    doc_ids = data["doc_ids"]
    embeddings = data["embeddings"]
    texts = data["texts"]  # ← جلب النصوص المنظفة

    query_vector = joblib.load(query_file)

    scores = cosine_similarity(embeddings, query_vector).flatten()
    ranked = sorted(zip(doc_ids, scores, texts), key=lambda x: x[1], reverse=True)
    return ranked
