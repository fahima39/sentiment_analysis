import pickle
import re

# ---------- Load Model & Vectorizer ----------
with open("../models/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("../models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ---------- Text Cleaning (same as training) ----------
def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------- Prediction Function ----------
def predict_sentiment(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    confidence = max(probability) * 100
    return prediction, confidence

# ---------- Test (run directly) ----------
if __name__ == "__main__":
    test_reviews = [
        "This movie was absolutely amazing, I loved it!",
        "Worst film I have ever seen, total waste of time.",
        "It was okay, nothing special but not bad either."
    ]

    for review in test_reviews:
        sentiment, confidence = predict_sentiment(review)
        print(f"Review: {review}")
        print(f"Prediction: {sentiment} (Confidence: {confidence:.2f}%)\n")