import pandas as pd
import re
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ---------- Step 1: Load Dataset ----------
print("Loading dataset...")
df = pd.read_csv("../data/IMDB Dataset.csv")
print(f"Dataset loaded: {df.shape[0]} rows")

# ---------- Step 2: Text Cleaning ----------
def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)          # remove HTML tags like <br />
    text = re.sub(r'[^a-zA-Z\s]', '', text)      # remove punctuation/numbers
    text = text.lower()                          # lowercase
    text = re.sub(r'\s+', ' ', text).strip()     # remove extra spaces
    return text

print("Cleaning text...")
df['cleaned_review'] = df['review'].apply(clean_text)

# ---------- Step 3: Feature Extraction (TF-IDF) ----------
print("Extracting features (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment']

# ---------- Step 4: Train-Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- Step 5: Train Model ----------
print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------- Step 6: Evaluate ----------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---------- Step 7: Save Model & Vectorizer ----------
os.makedirs("../models", exist_ok=True)

with open("../models/sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("../models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel and vectorizer saved in 'models/' folder!")