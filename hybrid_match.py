# hybrid_match.py
import os
import joblib
from sklearn.metrics.pairwise import cosine_similarity

REPRESENTATIONS_DIR = "representations"

def match_query_hybrid(table_name):
    hybrid_file = os.path.join(REPRESENTATIONS_DIR, f"hybrid_concat_embeddings_{table_name}.pkl")
    query_file = os.path.join(REPRESENTATIONS_DIR, "queries", f"hybrid_query_{table_name}.pkl")
    
    if not os.path.exists(hybrid_file) or not os.path.exists(query_file):
        raise FileNotFoundError("❌ لم يتم العثور على ملفات التمثيل الهجين المطلوبة.")
    
    data = joblib.load(hybrid_file)
    doc_ids = data["doc_ids"]
    embeddings = data["embeddings"]
    texts = data["texts"]  

    query_vector = joblib.load(query_file)

    scores = cosine_similarity(embeddings, query_vector).flatten()
    ranked = sorted(zip(doc_ids, scores, texts), key=lambda x: x[1], reverse=True)
    return ranked


