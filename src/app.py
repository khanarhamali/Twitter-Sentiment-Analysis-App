# src/app.py
import os
import joblib
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from src.preprocess import clean_text

# -----------------------------
# Model setup
# -----------------------------
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "RandomForest_pipeline.pkl")

# Dropbox direct download link
MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://www.dropbox.com/scl/fi/7a2imemlr6bzih4zk0del/RandomForest_pipeline.pkl?dl=1"
)

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Downloading model from {MODEL_URL}...")
    r = requests.get(MODEL_URL)
    if r.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("Model downloaded successfully.")
    else:
        raise Exception(f"Failed to download model, status code: {r.status_code}")

# Load pipeline
pipeline = joblib.load(MODEL_PATH)

# Reverse label mapping
reverse_mapping = {0: -1, 1: 0, 2: 1}

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Twitter Sentiment API")

class Tweet(BaseModel):
    text: str

@app.post("/predict")
def predict(tweet: Tweet):
    processed = clean_text(tweet.text)
    pred = pipeline.predict([processed])[0]
    label = reverse_mapping[pred]
    sentiment_map = {1:"Positive", 0:"Neutral", -1:"Negative"}
    return {"sentiment": label, "label": sentiment_map[label]}

@app.get("/")
def root():
    return {"status": "Twitter Sentiment API is running"}
