#hybrid.py
from flask import Flask, request, jsonify
import os
import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from database import fetch_cleaned_documents

app = Flask(__name__)

OUTPUT_DIR = "representations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_tfidf_matrix(table_name):
    matrix_path = os.path.join(OUTPUT_DIR, f"matrix_{table_name}.pkl")
    if not os.path.exists(matrix_path):
        return None, None, None
    data = joblib.load(matrix_path)
    return data["matrix"], data["doc_ids"], data["texts"]

def load_bert_embeddings(table_name):
    bert_path = os.path.join(OUTPUT_DIR, f"bert_embeddings_{table_name}.pkl")
    if not os.path.exists(bert_path):
        return None, None, None
    data = joblib.load(bert_path)
    return data["embeddings"], data["doc_ids"], data["texts"]

def reduce_tfidf_dim(tfidf_matrix, n_components=300):
    """
    تقليل أبعاد TF-IDF باستخدام SVD لتقليل حجم البيانات قبل الدمج.
    """
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(tfidf_matrix)
    return reduced

def build_and_save_hybrid_concat(table_name):
    hybrid_path = os.path.join(OUTPUT_DIR, f"hybrid_concat_embeddings_{table_name}.pkl")
    if os.path.exists(hybrid_path):
        return {"message": f"✅ تمثيل هجين بالتفرع لـ {table_name} محفوظ مسبقًا."}

    # تحميل تمثيل TF-IDF المحفوظ
    tfidf_matrix, tfidf_doc_ids, tfidf_texts = load_tfidf_matrix(table_name)
    if tfidf_matrix is None:
        return {"error": f"⚠️ ملفات TF-IDF لمجموعة {table_name} غير موجودة."}

    # تحميل تمثيل BERT المحفوظ
    bert_embeddings, bert_doc_ids, bert_texts = load_bert_embeddings(table_name)
    if bert_embeddings is None:
        return {"error": f"⚠️ ملفات BERT لمجموعة {table_name} غير موجودة."}

    # التحقق من تطابق doc_ids وترتيب النصوص
    if tfidf_doc_ids != bert_doc_ids:
        return {"error": "❌ عدم تطابق بين doc_ids في TF-IDF و BERT."}

    # تقليل أبعاد TF-IDF لتقليل حجم المصفوفة (مثلا 300 بعد)
    tfidf_reduced = reduce_tfidf_dim(tfidf_matrix, n_components=300)

    # دمج بالتفرع (concatenate)
    hybrid_embeddings = np.hstack((tfidf_reduced, bert_embeddings))

    # حفظ التمثيل الهجين بالتفرع
    joblib.dump({"doc_ids": tfidf_doc_ids, "embeddings": hybrid_embeddings, "texts": tfidf_texts}, hybrid_path)

    return {"message": f"✅ تم حفظ التمثيل الهجين بالتفرع لمجموعة {table_name} في {hybrid_path}"}


@app.route('/hybrid', methods=['POST'])
def build_hybrid_concat_endpoint():
    data = request.json
    table_name = data.get("table_name")

    if not table_name:
        return jsonify({"error": "الرجاء توفير 'table_name'"}), 400

    result = build_and_save_hybrid_concat(table_name)

    if "error" in result:
        return jsonify(result), 400

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)
