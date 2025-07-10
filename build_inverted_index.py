# # build_inverted_index.py
from flask import Flask, request, jsonify
import os
import pickle
import joblib

REPRESENTATIONS_DIR = "representations"

app = Flask(__name__)

def build_inverted_index(table_name):
    print(f"🚀 بناء الفهرس المعكوس لمجموعة: {table_name}")
    #ملف الفيكتورز لتمثيل tfidf المدرب على الوثائق 
    vectorizer_path = os.path.join(REPRESENTATIONS_DIR, f"vectorizer_{table_name}.pkl")
    #مصفوفة التمثيل للوثائق 
    matrix_path = os.path.join(REPRESENTATIONS_DIR, f"matrix_{table_name}.pkl")
    #مسار ملف الاندكس الخاص بالداتاسيت 
    index_path = os.path.join(REPRESENTATIONS_DIR, f"inverted_index_{table_name}.pkl")

    if os.path.exists(index_path):
        return {"message": f"✅ الفهرس المعكوس موجود مسبقًا: {index_path}"}
    #التحقق من وجود ملفات التمثيل tfidf 
    if not os.path.exists(vectorizer_path) or not os.path.exists(matrix_path):
        return {"error": "❌ ملفات التمثيل غير موجودة، يرجى توليد TF-IDF أولًا."}

    vectorizer = joblib.load(vectorizer_path)
    vocab = vectorizer.get_feature_names_out()

    data = joblib.load(matrix_path)
    tfidf_matrix = data["matrix"]
    doc_ids = data["doc_ids"]
    
    #بناء الفهرس هنا نحسب لكل مصطلح الوثائق التي تحتوي عليه 
    inverted_index = {}

    for term_idx, term in enumerate(vocab):
        docs_with_term = tfidf_matrix[:, term_idx].nonzero()[0]
        inverted_index[term] = [doc_ids[i] for i in docs_with_term]
    #حفظ الفهرس 
    with open(index_path, "wb") as f:
        pickle.dump(inverted_index, f)

    return {"message": f"✅ تم حفظ الفهرس المعكوس في: {index_path}"}


@app.route("/inverted_index", methods=["POST"])
def build_index_endpoint():
    data = request.json
    table_name = data.get("table_name")

    if not table_name:
        return jsonify({"error": "الرجاء توفير 'table_name'"}), 400

    try:
        result = build_inverted_index(table_name)
        if "error" in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"حدث خطأ غير متوقع: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004, debug=True)
