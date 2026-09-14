from flask import Flask, render_template, request
import pickle
import re
import os

app = Flask(__name__)

# ---------- Load Model & Vectorizer ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "models", "sentiment_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "models", "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

# ---------- Text Cleaning (same as training) ----------
def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------- Predict Function ----------
def predict_sentiment(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    confidence = round(max(probability) * 100, 2)
    return prediction, confidence

# ---------- Routes ----------
@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    confidence = None
    text = ""

    if request.method == "POST":
        text = request.form.get("text", "")

        if text.strip():
            sentiment, confidence = predict_sentiment(text)

    return render_template("index.html", sentiment=sentiment, confidence=confidence, text=text)


if __name__ == "__main__":
    app.run(debug=True)