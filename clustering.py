#clustering.py
import joblib
import os
from sklearn.cluster import KMeans
import numpy as np

REPRESENTATIONS_DIR = "representations"

def cluster_documents(table_name, representation, top_results=None, n_clusters=3):
    """
    يقوم بعمل clustering للوثائق باستخدام top_results.
    """

    if representation == "tfidf":
        file_path = os.path.join(REPRESENTATIONS_DIR, f"matrix_{table_name}.pkl")
    elif representation == "bert":
        file_path = os.path.join(REPRESENTATIONS_DIR, f"bert_embeddings_{table_name}.pkl")
    elif representation == "hybrid":
        file_path = os.path.join(REPRESENTATIONS_DIR, f"hybrid_concat_embeddings_{table_name}.pkl")
    else:
        raise ValueError(f"❌ تمثيل غير مدعوم: {representation}")
    #تحميل الداتا 
    data = joblib.load(file_path)

    if representation == "tfidf":
        embeddings = data["matrix"]
    else:
        embeddings = data["embeddings"]

    doc_ids = data["doc_ids"]
    texts = data["texts"]

    if top_results is not None:
        top_doc_ids = [doc_id for doc_id, _, _ in top_results]
        indices = [i for i, d in enumerate(doc_ids) if d in top_doc_ids]
        embeddings = embeddings[indices]
        doc_ids = [doc_ids[i] for i in indices]
        texts = [texts[i] for i in indices]
    # تنفيذ التجميع 
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    #انشاء قاموس المجموعات 
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        doc_info = {
            "doc_id": doc_ids[idx],
            "cleaned_text": texts[idx]
        }
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(doc_info)

    return clusters
