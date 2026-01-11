# src/predict.py
import os
import joblib
from src.preprocess import clean_text

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "RandomForest_pipeline.pkl")

# Load trained pipeline
pipeline = joblib.load(MODEL_PATH)

# Mapping back to -1, 0, 1
reverse_mapping = {0: -1, 1: 0, 2: 1}

# -----------------------------
# Negation handling
# -----------------------------
def handle_negations(text: str) -> str:
    """
    Convert negation + next word into single token.
    Example: 'not good' -> 'not_good'
    """
    words = text.split()
    new_words = []
    skip_next = False
    for i, w in enumerate(words):
        if skip_next:
            skip_next = False
            continue
        if w in {"not", "no", "never"} and i+1 < len(words):
            new_words.append(w + "_" + words[i+1])  # combine negation + next word
            skip_next = True
        else:
            new_words.append(w)
    return " ".join(new_words)

# -----------------------------
# Prediction function
# -----------------------------
def predict_sentiment(text: str) -> int:
    """
    Predict sentiment (-1, 0, 1) for input text.
    Handles negations and multi-word sentences.
    """
    processed = clean_text(text)
    processed = handle_negations(processed)  # <-- Apply negation tagging

    if not processed:
        return 0  # Default neutral

    pred = pipeline.predict([processed])[0]
    return reverse_mapping[pred]

# -----------------------------
# Test examples
# -----------------------------
if __name__ == "__main__":
    tests = [
        "I love Borderlands!",
        "This update is terrible",
        "Not bad, could be better",
        "good bad",
        "bad good",
        "I am not happy with this",
        "The movie was not good at all"
    ]

    for t in tests:
        print(t, "->", predict_sentiment(t))
