import pandas as pd
import nltk
import re
import pickle

nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Read datasets
fake_news = pd.read_csv("dataset/Fake.csv")
true_news = pd.read_csv("dataset/True.csv")

# Add labels
fake_news["label"] = 0
true_news["label"] = 1

# Merge datasets
news_data = pd.concat([fake_news, true_news])

# Shuffle dataset
news_data = news_data.sample(frac=1, random_state=42)

# Create content column
news_data["content"] = news_data["title"] + " " + news_data["text"]

# Text cleaning
def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    filtered_words = []

    for word in words:
        if word not in stop_words:
            filtered_words.append(word)

    return " ".join(filtered_words)

# Apply cleaning
news_data["content"] = news_data["content"].apply(clean_text)

# Input and Output
X = news_data["content"]
Y = news_data["label"]

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, Y_train)
# Save trained model
with open("models/model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save TF-IDF vectorizer
with open("models/vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

# Accuracy
Y_pred = model.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)

print()
print("==========================================")
print(" AI FAKE NEWS DETECTION SYSTEM ")
print("==========================================")
print()

print("Model Accuracy :", round(accuracy * 100, 2), "%")

print()

# User Input
user_news = input("Enter News Article : ")

# Clean input
user_news = clean_text(user_news)

# Convert to numbers
user_vector = vectorizer.transform([user_news])

# Prediction
prediction = model.predict(user_vector)

# Confidence Score
probability = model.predict_proba(user_vector)

confidence = max(probability[0]) * 100

print()

if prediction[0] == 0:
    print("Prediction : FAKE NEWS")
else:
    print("Prediction : TRUE NEWS")

print()

print("Confidence Score :", round(confidence, 2), "%")

print()

print("------------------------------------------")
print("Note:")
print()
print("• This prediction is generated using")
print("  a Machine Learning model.")
print()
print("• The model is trained on a")
print("  historical labeled news dataset.")
print()
print("• The prediction is based on")
print("  patterns learned from the")
print("  training data.")
print()
print("• Very recent or unseen news")
print("  articles may not always be")
print("  classified correctly.")
print()
print("• Final verification should")
print("  always be done using trusted")
print("  news sources.")
print("------------------------------------------")
print()
print("Model Saved Successfully")
print("Vectorizer Saved Successfully")