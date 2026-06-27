import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import (accuracy_score, f1_score, 
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import os

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

def evaluate():
    print("Evaluating model...")

    # Load test data
    test_df = pd.read_csv("data/processed/test.csv")
    test_df['clean_text'] = test_df['text'].apply(clean_text)

    # Load model and vectorizer
    model = joblib.load("backend/models/fakenews_model.pkl")
    vectorizer = joblib.load("backend/models/tfidf_vectorizer.pkl")

    X_test = vectorizer.transform(test_df['clean_text'])
    y_test = test_df['binary_label']
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nAccuracy : {acc*100:.2f}%")
    print(f"F1 Score : {f1:.4f}")
    print(f"\nClassification Report:\n")
    print(classification_report(y_test, y_pred, 
                                target_names=['FAKE', 'REAL']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['FAKE','REAL'],
                yticklabels=['FAKE','REAL'])
    plt.title('Confusion Matrix - ForensicAI')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    os.makedirs("report", exist_ok=True)
    plt.savefig("report/confusion_matrix.png")
    print("\nConfusion matrix saved to report/")
    plt.show()

if __name__ == "__main__":
    evaluate()