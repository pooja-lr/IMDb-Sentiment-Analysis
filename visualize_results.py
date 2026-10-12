import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


RESULTS_FILE = "outputs/model_comparison.csv"


def load_reviews(folder):
    reviews = []
    labels = []

    for sentiment, label in [("pos", 1), ("neg", 0)]:
        folder_path = os.path.join(folder, sentiment)

        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                with open(
                    os.path.join(folder_path, filename),
                    "r",
                    encoding="utf-8"
                ) as file:
                    reviews.append(file.read())

                labels.append(label)

    return reviews, labels


if not os.path.exists(RESULTS_FILE):
    raise SystemExit(
        "Results not found. Run python compare_models.py first."
    )

results = pd.read_csv(RESULTS_FILE)

os.makedirs("outputs", exist_ok=True)

# Chart 1: Compare model accuracy
plt.figure(figsize=(9, 5))
plt.bar(results["Model"], results["Accuracy"] * 100)
plt.title("IMDb Sentiment Classification - Model Accuracy")
plt.ylabel("Accuracy (%)")
plt.xlabel("Machine Learning Model")
plt.ylim(0, 100)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("outputs/model_accuracy.png", dpi=300)
plt.close()

# Chart 2: Compare F1-scores
plt.figure(figsize=(9, 5))
plt.bar(results["Model"], results["F1-Score"])
plt.title("IMDb Sentiment Classification - F1-Score")
plt.ylabel("F1-Score")
plt.xlabel("Machine Learning Model")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("outputs/model_f1_score.png", dpi=300)
plt.close()

print("Accuracy chart saved: outputs/model_accuracy.png")
print("F1-score chart saved: outputs/model_f1_score.png")

# Select the model with the highest measured F1-score
best_name = results.loc[
    results["F1-Score"].idxmax(), "Model"
]

model_options = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC()
}

print(f"\nBest model by F1-score: {best_name}")
print("Preparing its confusion matrix...")

X_train, y_train = load_reviews("aclImdb/train")
X_test, y_test = load_reviews("aclImdb/test")

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

best_model = model_options[best_name]
best_model.fit(X_train_tfidf, y_train)
predictions = best_model.predict(X_test_tfidf)

cm = confusion_matrix(y_test, predictions)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Positive"]
)

display.plot()
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=300)
plt.close()

print("Confusion matrix saved: outputs/confusion_matrix.png")
print("\nAll evaluation charts generated successfully!")