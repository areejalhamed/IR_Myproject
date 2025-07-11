# api_match.py
from flask import Flask, request, jsonify
from tfidf_match import match_query_tfidf
from bert_match import match_query_bert
from hybrid_match import match_query_hybrid

app = Flask(__name__)
@app.route('/query_match', methods=['POST'])
def match_query():
    data = request.json
    #قراءة الداتا الذي ادخلها المستخدم 
    #اسم الداتاسيت
    table_name = data.get("table_name")
    #طريقة التمثيل يلي اختارها اليوزر 
    representation = data.get("representation")
    if not table_name or not representation:
        return jsonify({"error": "يجب إرسال 'table_name' و 'representation'"}), 400
    try:
        if representation == "tfidf":
            results = match_query_tfidf(table_name)
        elif representation == "bert":
            results = match_query_bert(table_name)
        elif representation == "hybrid":
            results = match_query_hybrid(table_name)
        else:
            return jsonify({"error": "❌ تمثيل غير مدعوم"}), 400
        
        #ارجاع اول 10 نتائج فقط 
        top_results = results[:10]
        return jsonify({
            "message": "✅ تم الترتيب بنجاح.",
            "results": [{"doc_id": doc_id, "score": float(score), "text": text } for doc_id, score , text in top_results]
        })

    except Exception as e:
        return jsonify({"error": f"❌ خطأ: {str(e)}"}), 500
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5007, debug=True)
