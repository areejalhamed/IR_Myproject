# # hybrid_matching.py
# import os
# import joblib
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# REPRESENTATIONS_DIR = "representations"

# def match_query_hybrid(table_name):
#     tfidf_file = os.path.join(REPRESENTATIONS_DIR, f"matrix_{table_name}.pkl")
#     bert_file = os.path.join(REPRESENTATIONS_DIR, f"bert_embeddings_{table_name}.pkl")
#     tfidf_query_file = os.path.join(REPRESENTATIONS_DIR, "queries", f"tfidf_query_{table_name}.pkl")
#     bert_query_file = os.path.join(REPRESENTATIONS_DIR, "queries", f"bert_query_{table_name}.pkl")

#     if not (os.path.exists(tfidf_file) and os.path.exists(bert_file)
#             and os.path.exists(tfidf_query_file) and os.path.exists(bert_query_file)):
#         raise FileNotFoundError("❌ لم يتم العثور على بعض ملفات التمثيل المطلوبة.")

#     tfidf_data = joblib.load(tfidf_file)
#     bert_data = joblib.load(bert_file)

#     tfidf_matrix = tfidf_data["matrix"]
#     bert_matrix = np.asarray(bert_data["embeddings"])
#     doc_ids = bert_data["doc_ids"]

#     tfidf_query = np.asarray(joblib.load(tfidf_query_file))
#     bert_query = np.asarray(joblib.load(bert_query_file))

#     # ✅ طباعة الأشكال للتشخيص
#     print("📐 tfidf_matrix shape:", tfidf_matrix.shape)
#     print("📐 bert_matrix shape:", bert_matrix.shape)
#     print("📐 tfidf_query shape:", tfidf_query.shape)
#     print("📐 bert_query shape:", bert_query.shape)

#     if tfidf_matrix.ndim != 2 or bert_matrix.ndim != 2:
#         raise ValueError("❌ تم تحميل مصفوفة ليست ثنائية الأبعاد. تأكد من ملفات .pkl.")

#     # مطابقة الأبعاد
#     min_dim = min(tfidf_matrix.shape[1], bert_matrix.shape[1])
#     tfidf_matrix = tfidf_matrix[:, :min_dim]
#     bert_matrix = bert_matrix[:, :min_dim]
#     tfidf_query = tfidf_query[:, :min_dim]
#     bert_query = bert_query[:, :min_dim]

#     # تمثيل هجين
#     hybrid_matrix = np.asarray((tfidf_matrix + bert_matrix) / 2)
#     hybrid_query = np.asarray((tfidf_query + bert_query) / 2)


#     scores = cosine_similarity(hybrid_matrix, hybrid_query).flatten()
#     ranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)

#     return ranked

import os
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

REPRESENTATIONS_DIR = "representations"

def match_query_hybrid(table_name):
    tfidf_file = os.path.join(REPRESENTATIONS_DIR, f"matrix_{table_name}.pkl")
    bert_file = os.path.join(REPRESENTATIONS_DIR, f"bert_embeddings_{table_name}.pkl")
    tfidf_query_file = os.path.join(REPRESENTATIONS_DIR, "queries", f"tfidf_query_{table_name}.pkl")
    bert_query_file = os.path.join(REPRESENTATIONS_DIR, "queries", f"bert_query_{table_name}.pkl")

    if not (os.path.exists(tfidf_file) and os.path.exists(bert_file)
            and os.path.exists(tfidf_query_file) and os.path.exists(bert_query_file)):
        raise FileNotFoundError("❌ لم يتم العثور على بعض ملفات التمثيل المطلوبة.")

    tfidf_data = joblib.load(tfidf_file)
    bert_data = joblib.load(bert_file)

    tfidf_matrix = tfidf_data["matrix"]
    bert_matrix = np.asarray(bert_data["embeddings"])
    doc_ids = bert_data["doc_ids"]
    texts = bert_data["texts"]  # ← هنا أضف النصوص

    tfidf_query = np.asarray(joblib.load(tfidf_query_file))
    bert_query = np.asarray(joblib.load(bert_query_file))

    # ✅ طباعة الأشكال للتشخيص
    print("📐 tfidf_matrix shape:", tfidf_matrix.shape)
    print("📐 bert_matrix shape:", bert_matrix.shape)
    print("📐 tfidf_query shape:", tfidf_query.shape)
    print("📐 bert_query shape:", bert_query.shape)

    if tfidf_matrix.ndim != 2 or bert_matrix.ndim != 2:
        raise ValueError("❌ تم تحميل مصفوفة ليست ثنائية الأبعاد. تأكد من ملفات .pkl.")

    # مطابقة الأبعاد
    min_dim = min(tfidf_matrix.shape[1], bert_matrix.shape[1])
    tfidf_matrix = tfidf_matrix[:, :min_dim]
    bert_matrix = bert_matrix[:, :min_dim]
    tfidf_query = tfidf_query[:, :min_dim]
    bert_query = bert_query[:, :min_dim]

    # تمثيل هجين
    hybrid_matrix = np.asarray((tfidf_matrix + bert_matrix) / 2)
    hybrid_query = np.asarray((tfidf_query + bert_query) / 2)

    scores = cosine_similarity(hybrid_matrix, hybrid_query).flatten()
    ranked = sorted(zip(doc_ids, scores, texts), key=lambda x: x[1], reverse=True)

    return ranked
