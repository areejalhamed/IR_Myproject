#bert.py
from flask import Flask, request, jsonify
import os
import joblib
from sentence_transformers import SentenceTransformer #مكتبة لبناء تمثيلات نصية باستخدام البيرت 
from database import fetch_cleaned_documents

app = Flask(__name__)

OUTPUT_DIR = "representations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_and_save_bert(table_name, model_name='all-MiniLM-L6-v2'):
    output_path = os.path.join(OUTPUT_DIR, f"bert_embeddings_{table_name}.pkl")
    
    if os.path.exists(output_path):
        return {"message": f"✅ تمثيل BERT لمجموعة {table_name} محفوظ مسبقًا، تم التخطي."}
    
    #جلب المستندات المنظفة 
    doc_ids, texts = fetch_cleaned_documents(table_name)
    if not texts:
        return {"error": f"⚠️ مجموعة {table_name} فارغة، تم التخطي."}
    
    #يحمل نموذج bert 
    model = SentenceTransformer(model_name)
    #يرجعلي تمثيلات عددية لكل نص 
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    #حفظ البيانات 
    joblib.dump({"doc_ids": doc_ids, "embeddings": embeddings, "texts": texts}, output_path)

    return {"message": f"✅ تم حفظ ملفات BERT لمجموعة {table_name} في {output_path}"}

@app.route('/bert', methods=['POST'])
def build_bert_endpoint():
    data = request.json
    table_name = data.get("table_name")
    model_name = data.get("model_name", "all-MiniLM-L6-v2")

    if not table_name:
        return jsonify({"error": "الرجاء توفير 'table_name'"}), 400

    result = build_and_save_bert(table_name, model_name=model_name)

    if "error" in result:
        return jsonify(result), 400

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
