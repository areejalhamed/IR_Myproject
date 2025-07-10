#text_cleaner.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, jsonify

nltk.download('stopwords') #قائمة الكلمات الشائعة 
nltk.download('punkt') #تقسم الجمل الى كلمات (توكين)
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    #يقسم النص الى كلمات 
    tokens = nltk.word_tokenize(text)
    #يحذف الكلمات الشائعة 
    tokens = [word for word in tokens if word not in stop_words]
    #يحول الكلمات الى اصلها 
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join(tokens)

app = Flask(__name__)

@app.route('/clean_text', methods=['POST'])
def clean_text_api():
    data = request.json
    text = data.get("text", "")
    cleaned = clean_text(text)  # <-- يجب أن تكون الدالة معرفة هنا
    return jsonify({"cleaned_text": cleaned})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
