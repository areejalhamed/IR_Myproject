#tfidf.py
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer #تحويل النصوص لتمثيل عددي
import joblib
import os
from database import fetch_cleaned_documents

app = Flask(__name__)

#مجلد لحفظ ملفات التمثيلات العددية (فيكتورز و النصفوفة ) 
OUTPUT_DIR = "representations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#دالة لتوليد و حفظ ملفات التمثيل 
def vectorize_and_save(table_name):
    vectorizer_path = os.path.join(OUTPUT_DIR, f"vectorizer_{table_name}.pkl")
    matrix_path = os.path.join(OUTPUT_DIR, f"matrix_{table_name}.pkl")
    
    if os.path.exists(vectorizer_path) and os.path.exists(matrix_path):
        return {"message": f"✅ ملفات TF-IDF لمجموعة {table_name} موجودة مسبقًا، تم التخطي."}

    doc_ids, texts = fetch_cleaned_documents(table_name)
    if not texts:
        return {"error": f"⚠️ مجموعة {table_name} فارغة، تم التخطي."}

    vectorizer = TfidfVectorizer(max_features=5000)
    matrix = vectorizer.fit_transform(texts)

    joblib.dump(vectorizer, vectorizer_path)
    # 🟢 نضيف النصوص المنظفة في الحفظ
    joblib.dump({"doc_ids": doc_ids, "matrix": matrix, "texts": texts}, matrix_path)

    return {"message": f"✅ تم حفظ ملفات TF-IDF لمجموعة {table_name}"}


@app.route('/tfidf', methods=['POST'])
def build_tfidf_endpoint():
    data = request.json
    table_name = data.get("table_name")

    if not table_name:
        return jsonify({"error": "الرجاء توفير 'table_name'"}), 400

    result = vectorize_and_save(table_name)

    if "error" in result:
        return jsonify(result), 400

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


