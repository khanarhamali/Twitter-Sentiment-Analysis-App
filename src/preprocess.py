# src/preprocess.py
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# Safe NLTK loading
# -----------------------------
try:
    stop_words = set(stopwords.words("english"))
    negations = {"not", "no", "never"}  # Keep negation words
    stop_words = stop_words - negations
except LookupError:
    nltk.download("stopwords")
    nltk.download("wordnet")
    stop_words = set(stopwords.words("english"))
    negations = {"not", "no", "never"}
    stop_words = stop_words - negations

lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """Cleans input text for sentiment analysis."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+", "", text)        # Remove URLs
    text = re.sub(r"@\w+", "", text)           # Remove mentions
    text = re.sub(r"#", "", text)              # Remove hashtags symbol
    text = re.sub(rf"[{string.punctuation}\d]", " ", text)  # Remove punctuation & digits
    text = re.sub(r"\s+", " ", text).strip()   # Remove extra spaces

    tokens = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    ]

    return " ".join(tokens)
