# src/preprocess.py
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Safe NLTK loading
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    nltk.download("wordnet")
    stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(rf"[{string.punctuation}\d]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    ]

    return " ".join(tokens)
