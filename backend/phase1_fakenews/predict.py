import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = joblib.load("backend/models/fakenews_model.pkl")
vectorizer = joblib.load("backend/models/tfidf_vectorizer.pkl")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

def predict_news(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    confidence = abs(model.decision_function(vectorized)[0])
    confidence_pct = min(round(float(confidence) * 10, 2), 99.99)

    return {
        "prediction": "REAL" if prediction == 1 else "FAKE",
        "confidence": confidence_pct,
        "label": int(prediction)
    }

if __name__ == "__main__":
    test = "Scientists discover new vaccine that eliminates cancer completely"
    result = predict_news(test)
    print(f"Prediction: {result['prediction']} | Confidence: {result['confidence']}%")