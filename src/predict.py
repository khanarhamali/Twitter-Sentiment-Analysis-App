# src/predict.py
import os
import joblib
from src.preprocess import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "RandomForest_pipeline.pkl")

pipeline = joblib.load(MODEL_PATH)

reverse_mapping = {0: -1, 1: 0, 2: 1}

def predict_sentiment(text: str) -> int:
    processed = clean_text(text)
    pred = pipeline.predict([processed])[0]
    return reverse_mapping[pred]

if __name__ == "__main__":
    tests = [
        "I love Borderlands!",
        "This update is terrible",
        "Not bad, could be better"
    ]

    for t in tests:
        print(t, "->", predict_sentiment(t))
