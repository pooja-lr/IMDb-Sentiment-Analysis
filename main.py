import os
import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def load_reviews(folder):
    reviews = []

    for sentiment in ["pos", "neg"]:
        folder_path = os.path.join(folder, sentiment)
        label = 1 if sentiment == "pos" else 0

        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)

                with open(file_path, "r", encoding="utf-8") as file:
                    review = file.read()

                reviews.append({
                    "review": review,
                    "sentiment": label
                })

    return pd.DataFrame(reviews)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


if __name__ == "__main__":
    print("Loading IMDb dataset...")

    train_data = load_reviews(os.path.join("aclImdb", "train"))
    test_data = load_reviews(os.path.join("aclImdb", "test"))

    print("Training reviews:", len(train_data))
    print("Testing reviews:", len(test_data))

    print("\nPreprocessing reviews...")

    train_data["review"] = train_data["review"].apply(clean_text)
    test_data["review"] = test_data["review"].apply(clean_text)

    X_train = train_data["review"]
    y_train = train_data["sentiment"]

    X_test = test_data["review"]
    y_test = test_data["sentiment"]

    print("\nTraining Logistic Regression model...")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=2
        )),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    print("\nEvaluating model...")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Negative", "Positive"]
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    import joblib
    joblib.dump(model, os.path.join("models", "sentiment_model.pkl"))

    print("\nModel saved successfully!")
    print("Location: models/sentiment_model.pkl")