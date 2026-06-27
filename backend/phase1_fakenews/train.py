import pandas as pd
from preprocess import preprocess
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os

def train_models():
    print("Training models...")

    X_train, X_test, y_train, y_test, _ = preprocess(save_vectorizer=False)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Passive Aggressive": PassiveAggressiveClassifier(max_iter=50),
        "Naive Bayes": MultinomialNB()
    }

    best_model = None
    best_score = 0
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        results[name] = {"accuracy": round(acc*100, 2), "f1": round(f1, 4)}
        print(f"{name} → Accuracy: {acc*100:.2f}% | F1: {f1:.4f}")

        if acc > best_score:
            best_score = acc
            best_model = model
            best_name = name

    # Save best model
    os.makedirs("backend/models", exist_ok=True)
    joblib.dump(best_model, "backend/models/fakenews_model.pkl")
    print(f"\nBest Model: {best_name} ({best_score*100:.2f}%)")
    print("Model saved to backend/models/")

    return results

if __name__ == "__main__":
    train_models()