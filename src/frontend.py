# src/frontend.py
import streamlit as st
import requests

# For local machine testing
API_URL = "http://127.0.0.1:8000/predict"


st.set_page_config(page_title="Twitter Sentiment", layout="centered")

st.title("🐦 Twitter Sentiment Analyzer")
st.write("Enter a tweet and get its sentiment")

text = st.text_area("Enter your text here:")

if st.button("Analyze Sentiment"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        response = requests.post(
            API_URL,
            json={"text": text}
        )

        if response.status_code == 200:
            sentiment = response.json()["sentiment"]

            if sentiment == 1:
                st.success("😊 Positive")
            elif sentiment == 0:
                st.info("😐 Neutral")
            else:
                st.error("😠 Negative")
        else:
            st.error("API Error")
