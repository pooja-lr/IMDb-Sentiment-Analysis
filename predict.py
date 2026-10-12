import joblib
import os

MODEL_PATH = os.path.join("models", "sentiment_model.pkl")

if not os.path.exists(MODEL_PATH):
    print("Model not found. Please run python main.py first.")
    raise SystemExit(1)

model = joblib.load(MODEL_PATH)

print("=" * 45)
print("     IMDb SENTIMENT ANALYSIS")
print("=" * 45)
print("Enter a movie review to analyze its sentiment.")
print("Type 'exit' to quit.\n")

while True:
    review = input("Enter your review: ").strip()

    if review.lower() == "exit":
        print("Thank you for using IMDb Sentiment Analysis!")
        break

    if not review:
        print("Please enter a non-empty review.\n")
        continue

    prediction = model.predict([review])[0]

    sentiment = "Positive" if prediction == 1 else "Negative"

    print(f"Predicted Sentiment: {sentiment}\n")
