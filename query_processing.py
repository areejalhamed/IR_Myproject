#query_processing.py
import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from text_cleaner import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

app = Flask(__name__)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

REPRESENTATIONS_DIR = "representations"
QUERY_REPRESENTATIONS_DIR = os.path.join(REPRESENTATIONS_DIR, "queries")
os.makedirs(QUERY_REPRESENTATIONS_DIR, exist_ok=True)

@app.route("/process_query", methods=["POST"])
def process_query():
    data = request.get_json()
    query_text = data.get("query")
    table_name = data.get("table_name")
    representation = data.get("representation")

    if not query_text or not table_name or not representation:
        return jsonify({"error": "يرجى إرسال 'query', 'table_name', و 'representation'"}), 400

    try:
        cleaned_query = clean_text(query_text)
        result_files = {}

        # -------- TF-IDF فقط --------
        if representation == "tfidf":
            vectorizer_path = os.path.join(REPRESENTATIONS_DIR, f"vectorizer_{table_name}.pkl")
            if not os.path.exists(vectorizer_path):
                return jsonify({"error": f"❌ لم يتم العثور على vectorizer الخاص بـ {table_name}"}), 400
            
            vectorizer = joblib.load(vectorizer_path)
            tfidf_vector = vectorizer.transform([cleaned_query])
            tfidf_vector = normalize(tfidf_vector, axis=1)
            tfidf_vector_array = tfidf_vector.toarray()

            tfidf_file = os.path.join(QUERY_REPRESENTATIONS_DIR, f"tfidf_query_{table_name}.pkl")
            joblib.dump(tfidf_vector_array, tfidf_file)
            result_files["tfidf"] = tfidf_file

        # -------- BERT فقط --------
        elif representation == "bert":
            bert_vector = model.encode([cleaned_query], normalize_embeddings=True)
            bert_file = os.path.join(QUERY_REPRESENTATIONS_DIR, f"bert_query_{table_name}.pkl")
            joblib.dump(bert_vector, bert_file)
            result_files["bert"] = bert_file

        # -------- Hybrid فقط --------
        elif representation == "hybrid":
            vectorizer_path = os.path.join(REPRESENTATIONS_DIR, f"vectorizer_{table_name}.pkl")
            matrix_path = os.path.join(REPRESENTATIONS_DIR, f"matrix_{table_name}.pkl")

            if not os.path.exists(vectorizer_path):
                return jsonify({"error": f"❌ لم يتم العثور على vectorizer الخاص بـ {table_name}"}), 400
            if not os.path.exists(matrix_path):
                return jsonify({"error": f"❌ ملف المصفوفة TF-IDF {table_name} غير موجود لتطبيق SVD"}), 400

            vectorizer = joblib.load(vectorizer_path)
            data_matrix = joblib.load(matrix_path)
            tfidf_matrix_docs = data_matrix["matrix"]

            tfidf_vector = vectorizer.transform([cleaned_query])
            tfidf_vector = normalize(tfidf_vector, axis=1)
            tfidf_vector_array = tfidf_vector.toarray()

            svd = TruncatedSVD(n_components=300, random_state=42)
            svd.fit(tfidf_matrix_docs)
            tfidf_vector_reduced = svd.transform(tfidf_vector_array)

            bert_vector = model.encode([cleaned_query], normalize_embeddings=True)

            hybrid_vector = np.hstack((tfidf_vector_reduced, bert_vector))
            hybrid_file = os.path.join(QUERY_REPRESENTATIONS_DIR, f"hybrid_query_{table_name}.pkl")
            joblib.dump(hybrid_vector, hybrid_file)
            result_files["hybrid"] = hybrid_file

        else:
            return jsonify({"error": "❌ التمثيل المحدد غير مدعوم. اختر 'tfidf' أو 'bert' أو 'hybrid'."}), 400

        return jsonify({
            "message": "✅ تمت معالجة الاستعلام بنجاح.",
            "query_cleaned": cleaned_query,
            "files": result_files
        })

    except Exception as e:
        return jsonify({"error": f"❌ خطأ أثناء المعالجة: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=True)

