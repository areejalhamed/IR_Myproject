
# hybrid.py
from flask import Flask, request, jsonify
import os
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

REPRESENTATIONS_DIR = "representations"

def load_document_representations(table_name, method):
    if method == "tfidf":
        # استبدال pickle.load بـ joblib.load لتوافق مع طريقة الحفظ
        data = joblib.load(os.path.join(REPRESENTATIONS_DIR, f"matrix_{table_name}.pkl"))
        matrix = data["matrix"]
    elif method == "bert":
        data = joblib.load(os.path.join(REPRESENTATIONS_DIR, f"bert_embeddings_{table_name}.pkl"))
        matrix = data["embeddings"]
    else:
        raise ValueError("Unknown method")

    normalized_matrix = normalize(matrix, axis=1)
    return data["doc_ids"], normalized_matrix

def get_query_embedding(query_text, table_name, method):
    if method == "tfidf":
        vectorizer_path = os.path.join(REPRESENTATIONS_DIR, f"vectorizer_{table_name}.pkl")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"❌ لم يتم العثور على ملف الـ vectorizer: {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)
        vector = vectorizer.transform([query_text])
        vector = normalize(vector, axis=1)
        return vector

    elif method == "bert":
        model_path = os.path.join(REPRESENTATIONS_DIR, "bert_model_cache")
        os.makedirs(model_path, exist_ok=True)

        model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=model_path)
        embedding = model.encode([query_text], normalize_embeddings=True)
        return embedding

    else:
        raise ValueError("Unknown method")

def run_search(query_text, target_table):
    tfidf_path = os.path.join(REPRESENTATIONS_DIR, f"matrix_{target_table}.pkl")
    bert_path = os.path.join(REPRESENTATIONS_DIR, f"bert_embeddings_{target_table}.pkl")

    if not os.path.exists(tfidf_path) or not os.path.exists(bert_path):
        raise FileNotFoundError("❌ ملفات تمثيلات الوثائق غير موجودة، يرجى تشغيل ملفات البناء أولاً.")

    doc_ids, doc_tfidf = load_document_representations(target_table, "tfidf")
    _, doc_bert = load_document_representations(target_table, "bert")

    query_tfidf = get_query_embedding(query_text, target_table, "tfidf")
    query_bert = get_query_embedding(query_text, target_table, "bert")

    sim_tfidf = cosine_similarity(query_tfidf, doc_tfidf)
    sim_bert = cosine_similarity(query_bert, doc_bert)

    fusion_scores = (sim_tfidf + sim_bert) / 2
    ranked_indices = fusion_scores[0].argsort()[::-1]

    results = []
    for idx in ranked_indices:
        results.append({
            "doc_id": doc_ids[idx],
            "score": float(fusion_scores[0][idx])
        })

    return results

@app.route('/hybrid', methods=['POST'])
def hybrid_search_api():
    data = request.json
    query = data.get("query")
    table_name = data.get("table_name")

    if not query or not table_name:
        return jsonify({"error": "الرجاء توفير 'query' و 'table_name'"}), 400

    try:
        results = run_search(query, table_name)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"حدث خطأ غير متوقع: {str(e)}"}), 500

    return jsonify({
        "message": "✅ تم البحث الهجين بنجاح",
        "results": results
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)
