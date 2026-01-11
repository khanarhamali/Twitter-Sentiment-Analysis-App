# src/app.py
import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

from src.preprocess import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "RandomForest_pipeline.pkl")

pipeline = joblib.load(MODEL_PATH)

reverse_mapping = {0: -1, 1: 0, 2: 1}

app = FastAPI(title="Twitter Sentiment API")

class Tweet(BaseModel):
    text: str

@app.post("/predict")
def predict(tweet: Tweet):
    processed = clean_text(tweet.text)
    pred = pipeline.predict([processed])[0]
    label = reverse_mapping[pred]

    sentiment_map = {
        1: "Positive",
        0: "Neutral",
        -1: "Negative"
    }

    return {
        "sentiment": label,
        "label": sentiment_map[label]
    }

@app.get("/")
def root():
    return {"status": "Twitter Sentiment API is running"}
