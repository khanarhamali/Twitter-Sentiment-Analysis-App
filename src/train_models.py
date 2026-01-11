# src/train_models.py
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.preprocess import clean_text

# -----------------------------
# Paths (ABSOLUTE & SAFE)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "twitter_training.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "RandomForest_pipeline.pkl")

os.makedirs(MODELS_DIR, exist_ok=True)

def train():
    # Load data
    df = pd.read_csv(DATA_PATH)

    df = df.rename(columns={
        'im getting on borderlands and i will murder you all ,': 'Comments',
        'Positive': 'Sentiments'
    })

    df = df.drop(df[df["Sentiments"] == "Irrelevant"].index)
    df = df.dropna(subset=["Comments"])

    df["Sentiments"] = df["Sentiments"].map({
        "Negative": -1,
        "Neutral": 0,
        "Positive": 1
    })

    df["Comments"] = df["Comments"].apply(clean_text)

    X = df["Comments"]
    y = df["Sentiments"].map({-1: 0, 0: 1, 1: 2})  # 0-based labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = ImbPipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("smote", SMOTE(random_state=42)),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, MODEL_PATH)
    print(f"✅ Model trained & saved at: {MODEL_PATH}")

if __name__ == "__main__":
    train()
