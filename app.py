# #app.py
# from flask import Flask, request, jsonify, send_from_directory
# import os, joblib, csv, pickle
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import normalize
# from sentence_transformers import SentenceTransformer, util
# from spellchecker import SpellChecker
# import nltk
# from nltk.corpus import wordnet
# from text_cleaner import clean_text
# from clustering import cluster_documents

# nltk.download('wordnet')
# nltk.download('omw-1.4')

# app = Flask(__name__)

# REPRESENTATIONS_DIR = "representations"
# QUERY_REPRESENTATIONS_DIR = os.path.join(REPRESENTATIONS_DIR, "queries")
# os.makedirs(QUERY_REPRESENTATIONS_DIR, exist_ok=True)

# model = SentenceTransformer('all-MiniLM-L6-v2')
# spell = SpellChecker(language='en')

# @app.route("/")
# def serve_interface():
#     return send_from_directory(".", "interface.html")

# def correct_spelling(query):
#     return ' '.join([spell.correction(w) or w for w in query.split()])

# def suggest_similar_queries(query, all_docs, top_k=4):
#     q_vec = model.encode(query, convert_to_tensor=True)
#     scores = util.cos_sim(q_vec, all_docs)[0]
#     top = scores.topk(k=top_k)
#     return [all_docs[i] for i in top.indices]

# def expand_query(query):
#     terms = set(query.split())
#     for word in query.split():
#         for syn in wordnet.synsets(word):
#             for lemma in syn.lemmas():
#                 terms.add(lemma.name().replace('_', ' '))
#     return list(terms)

# @app.route("/refine_query", methods=["POST"])
# def refine_query():
#     data = request.get_json()
#     user_query = data["query"]

#     corrected = correct_spelling(user_query)
#     expanded = expand_query(corrected)

#     return jsonify({
#         "corrected_query": corrected,
#         "similar_queries": [],  # تم التعليق مؤقتًا حتى لا نخلط بالنتائج القديمة
#         "expanded_terms": expanded
#     })

# @app.route("/process_query", methods=["POST"])
# def process_query():
#     data = request.get_json()
#     query_text = data.get("query")
#     table_name = data.get("table_name")
#     representation = data.get("representation")

#     if not query_text or not table_name or not representation:
#         return jsonify({"error": "يرجى إرسال 'query', 'table_name', و 'representation'"}), 400

#     try:
#         cleaned_query = clean_text(query_text)
#         result_files = {}

#         if representation in ("tfidf", "hybrid"):
#             vectorizer_path = os.path.join(REPRESENTATIONS_DIR, f"vectorizer_{table_name}.pkl")
#             if not os.path.exists(vectorizer_path):
#                 return jsonify({"error": f"❌ لم يتم العثور على vectorizer الخاص بـ {table_name}"}), 400
#             vectorizer = joblib.load(vectorizer_path)
#             tfidf_vector = vectorizer.transform([cleaned_query])
#             tfidf_vector = normalize(tfidf_vector, axis=1)
#             tfidf_vector_array = tfidf_vector.toarray()
#             tfidf_file = os.path.join(QUERY_REPRESENTATIONS_DIR, f"tfidf_query_{table_name}.pkl")
#             joblib.dump(tfidf_vector_array, tfidf_file)
#             result_files["tfidf"] = tfidf_file

#         if representation in ("bert", "hybrid"):
#             bert_vector = model.encode([cleaned_query], normalize_embeddings=True)
#             bert_file = os.path.join(QUERY_REPRESENTATIONS_DIR, f"bert_query_{table_name}.pkl")
#             joblib.dump(bert_vector, bert_file)
#             result_files["bert"] = bert_file

#         if representation == "hybrid":
#             min_dim = min(tfidf_vector_array.shape[1], bert_vector.shape[1])
#             hybrid_vector = (tfidf_vector_array[:, :min_dim] + bert_vector[:, :min_dim]) / 2
#             hybrid_file = os.path.join(QUERY_REPRESENTATIONS_DIR, f"hybrid_query_{table_name}.pkl")
#             joblib.dump(hybrid_vector, hybrid_file)
#             result_files["hybrid"] = hybrid_file

#         return jsonify({
#             "message": "✅ تمت معالجة الاستعلام بنجاح.",
#             "query_cleaned": cleaned_query,
#             "files": result_files
#         })

#     except Exception as e:
#         return jsonify({"error": f"❌ خطأ أثناء المعالجة: {str(e)}"}), 500

# @app.route("/query_match", methods=["POST"])
# def match_query():
#     data = request.get_json()
#     table_name = data.get("table_name")
#     representation = data.get("representation")
#     query_text = data.get("query")
#     mode = data.get("mode", "basic")

#     top_k = 10

#     try:
#         if representation == "tfidf":
#             results = match_query_tfidf(table_name)
#         elif representation == "bert":
#             results = match_query_bert(table_name)
#         elif representation == "hybrid":
#             results = match_query_hybrid(table_name)
#         else:
#             return jsonify({"error": "❌ تمثيل غير مدعوم"}), 400

#         # إعداد نتائج للتجميع (top 100)
#         results_for_clustering = results[:100]

#         clusters = None
#         if mode == "basic_extra":
#             clusters_dict = cluster_documents(table_name, representation, top_results=results_for_clustering, n_clusters=5)
#             clusters = []
#             for label, docs in clusters_dict.items():
#                 clusters.append({
#                     "cluster_id": int(label),
#                     "documents": [{"doc_id": int(doc["doc_id"]), "cleaned_text": doc["cleaned_text"]} for doc in docs]
#                 })

#         results_json = [{"doc_id": int(doc_id), "score": float(score), "cleaned_text": text} for doc_id, score, text in results[:top_k]]

#         return jsonify({
#             "message": "✅ تم الترتيب بنجاح.",
#             "results": results_json,
#             "clusters": clusters
#         })

#     except Exception as e:
#         return jsonify({"error": f"❌ خطأ: {str(e)}"}), 500

# def match_query_tfidf(table_name):
#     matrix_file = os.path.join(REPRESENTATIONS_DIR, f"matrix_{table_name}.pkl")
#     query_file = os.path.join(REPRESENTATIONS_DIR, "queries", f"tfidf_query_{table_name}.pkl")
#     data = joblib.load(matrix_file)
#     matrix = data["matrix"]
#     doc_ids = data["doc_ids"]
#     texts = data["texts"]

#     query_vector = joblib.load(query_file)

#     scores = cosine_similarity(matrix, query_vector).flatten()
#     ranked = sorted(zip(doc_ids, scores, texts), key=lambda x: x[1], reverse=True)
#     return ranked

# def match_query_bert(table_name):
#     bert_file = os.path.join(REPRESENTATIONS_DIR, f"bert_embeddings_{table_name}.pkl")
#     query_file = os.path.join(REPRESENTATIONS_DIR, "queries", f"bert_query_{table_name}.pkl")
#     data = joblib.load(bert_file)
#     embeddings = data["embeddings"]
#     doc_ids = data["doc_ids"]
#     texts = data["texts"]

#     query_vector = joblib.load(query_file)

#     scores = cosine_similarity(embeddings, query_vector).flatten()
#     ranked = sorted(zip(doc_ids, scores, texts), key=lambda x: x[1], reverse=True)
#     return ranked

# def match_query_hybrid(table_name):
#     hybrid_file = os.path.join(REPRESENTATIONS_DIR, f"hybrid_embeddings_{table_name}.pkl")
#     query_file = os.path.join(REPRESENTATIONS_DIR, "queries", f"hybrid_query_{table_name}.pkl")
#     data = joblib.load(hybrid_file)
#     embeddings = data["embeddings"]
#     doc_ids = data["doc_ids"]
#     texts = data["texts"]

#     query_vector = joblib.load(query_file)

#     scores = cosine_similarity(embeddings, query_vector).flatten()
#     ranked = sorted(zip(doc_ids, scores, texts), key=lambda x: x[1], reverse=True)
#     return ranked

# if __name__ == "__main__":
#     app.run(debug=True, port=5000)


# app.py
from flask import Flask, request, jsonify, send_from_directory
import os, joblib, pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer, util
from spellchecker import SpellChecker
import nltk
from nltk.corpus import wordnet
from text_cleaner import clean_text
from clustering import cluster_documents

nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__)

REPRESENTATIONS_DIR = "representations"
QUERY_REPRESENTATIONS_DIR = os.path.join(REPRESENTATIONS_DIR, "queries")
os.makedirs(QUERY_REPRESENTATIONS_DIR, exist_ok=True)

model = SentenceTransformer('all-MiniLM-L6-v2')
spell = SpellChecker(language='en')

@app.route("/")
def serve_interface():
    return send_from_directory(".", "interface.html")

def correct_spelling(query):
    return ' '.join([spell.correction(w) or w for w in query.split()])

def suggest_similar_queries(query, all_docs, top_k=4):
    q_vec = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(q_vec, all_docs)[0]
    top = scores.topk(k=top_k)
    return [all_docs[i] for i in top.indices]

def expand_query(query):
    terms = set(query.split())
    for word in query.split():
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                terms.add(lemma.name().replace('_', ' '))
    return list(terms)

@app.route("/refine_query", methods=["POST"])
def refine_query():
    data = request.get_json()
    user_query = data["query"]

    corrected = correct_spelling(user_query)
    expanded = expand_query(corrected)

    return jsonify({
        "corrected_query": corrected,
        "similar_queries": [],
        "expanded_terms": expanded
    })

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

        if representation in ("tfidf", "hybrid"):
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

        if representation in ("bert", "hybrid"):
            bert_vector = model.encode([cleaned_query], normalize_embeddings=True)
            bert_file = os.path.join(QUERY_REPRESENTATIONS_DIR, f"bert_query_{table_name}.pkl")
            joblib.dump(bert_vector, bert_file)
            result_files["bert"] = bert_file

        if representation == "hybrid":
            min_dim = min(tfidf_vector_array.shape[1], bert_vector.shape[1])
            hybrid_vector = (tfidf_vector_array[:, :min_dim] + bert_vector[:, :min_dim]) / 2
            hybrid_file = os.path.join(QUERY_REPRESENTATIONS_DIR, f"hybrid_query_{table_name}.pkl")
            joblib.dump(hybrid_vector, hybrid_file)
            result_files["hybrid"] = hybrid_file

        return jsonify({
            "message": "✅ تمت معالجة الاستعلام بنجاح.",
            "query_cleaned": cleaned_query,
            "files": result_files
        })

    except Exception as e:
        return jsonify({"error": f"❌ خطأ أثناء المعالجة: {str(e)}"}), 500

@app.route("/query_match", methods=["POST"])
def match_query():
    data = request.get_json()
    table_name = data.get("table_name")
    representation = data.get("representation")
    query_text = data.get("query")
    mode = data.get("mode", "basic")

    top_k = 10

    try:
        if representation == "tfidf":
            results = match_query_tfidf(table_name)
        elif representation == "bert":
            results = match_query_bert(table_name)
        elif representation == "hybrid":
            results = match_query_hybrid(table_name)
        else:
            return jsonify({"error": "❌ تمثيل غير مدعوم"}), 400

        results_for_clustering = results[:100]

        clusters = None
        if mode == "basic_extra":
            clusters_dict = cluster_documents(table_name, representation, top_results=results_for_clustering, n_clusters=5)
            clusters = []
            for label, docs in clusters_dict.items():
                clusters.append({
                    "cluster_id": int(label),
                    "documents": [{"doc_id": int(doc["doc_id"]), "cleaned_text": doc["cleaned_text"]} for doc in docs]
                })

        results_json = [{"doc_id": int(doc_id), "score": float(score), "cleaned_text": text} for doc_id, score, text in results[:top_k]]

        return jsonify({
            "message": "✅ تم الترتيب بنجاح.",
            "results": results_json,
            "clusters": clusters
        })

    except Exception as e:
        return jsonify({"error": f"❌ خطأ: {str(e)}"}), 500

def match_query_tfidf(table_name):
    matrix_file = os.path.join(REPRESENTATIONS_DIR, f"matrix_{table_name}.pkl")
    query_file = os.path.join(REPRESENTATIONS_DIR, "queries", f"tfidf_query_{table_name}.pkl")
    index_file = os.path.join(REPRESENTATIONS_DIR, f"inverted_index_{table_name}.pkl")
    vectorizer_path = os.path.join(REPRESENTATIONS_DIR, f"vectorizer_{table_name}.pkl")

    data = joblib.load(matrix_file)
    matrix = data["matrix"]
    doc_ids = data["doc_ids"]
    texts = data["texts"]

    query_vector = joblib.load(query_file)

    if not os.path.exists(index_file):
        raise FileNotFoundError(f"❌ لم يتم العثور على الفهرس المعكوس: {index_file}")

    inverted_index = joblib.load(index_file)
    vectorizer = joblib.load(vectorizer_path)
    feature_names = vectorizer.get_feature_names_out()

    query_terms_indices = query_vector.nonzero()[1]
    terms_in_query = [feature_names[i] for i in query_terms_indices if i < len(feature_names)]

    candidate_doc_ids = set()
    for term in terms_in_query:
        if term in inverted_index:
            candidate_doc_ids.update(inverted_index[term])

    if not candidate_doc_ids:
        candidate_doc_ids = set(doc_ids)

    candidate_indices = [i for i, doc_id in enumerate(doc_ids) if doc_id in candidate_doc_ids]
    reduced_matrix = matrix[candidate_indices]

    scores = cosine_similarity(reduced_matrix, query_vector).flatten()

    ranked_candidates = sorted(
        zip([doc_ids[i] for i in candidate_indices], scores, [texts[i] for i in candidate_indices]),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked_candidates

def match_query_bert(table_name):
    bert_file = os.path.join(REPRESENTATIONS_DIR, f"bert_embeddings_{table_name}.pkl")
    query_file = os.path.join(REPRESENTATIONS_DIR, "queries", f"bert_query_{table_name}.pkl")
    data = joblib.load(bert_file)
    embeddings = data["embeddings"]
    doc_ids = data["doc_ids"]
    texts = data["texts"]

    query_vector = joblib.load(query_file)

    scores = cosine_similarity(embeddings, query_vector).flatten()
    ranked = sorted(zip(doc_ids, scores, texts), key=lambda x: x[1], reverse=True)
    return ranked

def match_query_hybrid(table_name):
    hybrid_file = os.path.join(REPRESENTATIONS_DIR, f"hybrid_embeddings_{table_name}.pkl")
    query_file = os.path.join(REPRESENTATIONS_DIR, "queries", f"hybrid_query_{table_name}.pkl")
    data = joblib.load(hybrid_file)
    embeddings = data["embeddings"]
    doc_ids = data["doc_ids"]
    texts = data["texts"]

    query_vector = joblib.load(query_file)

    scores = cosine_similarity(embeddings, query_vector).flatten()
    ranked = sorted(zip(doc_ids, scores, texts), key=lambda x: x[1], reverse=True)
    return ranked

if __name__ == "__main__":
    app.run(debug=True, port=5000) 