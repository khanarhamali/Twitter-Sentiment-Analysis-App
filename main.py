# main.py
from src.train_models import train
from src.predict import predict_sentiment
import os

# ------------------------
# Step 1: Train the model (sirf jab zaroorat ho)
# train() already saves model internally
# Uncomment if you want to retrain
# train()

# ------------------------
# Step 2: Test prediction
test_text = "I am so happy with this game!"
print("Test Prediction:", predict_sentiment(test_text))
