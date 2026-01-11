# main.py
from src.train_models import train
from src.predict import predict_sentiment

# Train model (sirf jab zaroorat ho)
# train()

print(predict_sentiment("I am so happy with this game!"))
