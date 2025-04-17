from flask import Flask, render_template, request
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Clean input message
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    return ' '.join(tokens)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    cleaned = clean_text(message)
    vect_msg = vectorizer.transform([cleaned])
    prediction = model.predict(vect_msg)[0]
    result = "🚫 Spam" if prediction == 1 else "✅ Not Spam"
    return render_template('index.html', prediction=result, message=message)

if __name__ == '__main__':
    app.run(debug=True)
