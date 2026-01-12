# Twitter Sentiment Analysis App 🚀

A complete **end-to-end Machine Learning project** for Twitter sentiment analysis, covering **data preprocessing, model training, experiment tracking, FastAPI backend, Streamlit frontend, Dockerization, and CI/CD using GitHub Actions**.

---

## 📌 Project Overview

This project analyzes tweet text and predicts sentiment as:

* **Positive** 😊
* **Neutral** 😐
* **Negative** ☹️

It is designed following **industry-level MLOps practices**, making it suitable for learning, demonstration, and portfolio purposes.

---

## 🧠 Machine Learning Details

### 🔹 Model Used

* **Random Forest Classifier**
* Trained using **scikit-learn**
* Full pipeline saved using **joblib** (`RandomForest_pipeline.pkl`)

### 🔹 Why Random Forest?

* Handles non-linear relationships well
* Robust to overfitting
* Performs well on text-based features

---

## 🔧 Preprocessing Techniques

Text preprocessing is handled in `src/preprocess.py` using **NLTK**:

* Lowercasing
* URL removal
* Special character removal
* Tokenization
* Stopword removal
* Lemmatization

This ensures clean and consistent input for the ML model.

---

## 🧪 Experiment Tracking (MLflow)

MLflow is used for:

* Tracking experiments
* Logging metrics (accuracy, precision, recall, F1-score)
* Comparing multiple models

> ⚠️ MLflow is used locally for experimentation and is **not required during deployment**.

---

## ⚙️ Backend (FastAPI)

### 🔹 Features

* High-performance REST API
* Automatic Swagger documentation
* JSON-based input/output

### 🔹 Endpoints

| Method | Endpoint   | Description             |
| ------ | ---------- | ----------------------- |
| GET    | `/`        | Health check            |
| POST   | `/predict` | Predict tweet sentiment |

Example request:

```json
{
  "text": "I love this game!"
}
```

---

## 🎨 Frontend (Streamlit)

* Simple UI for user interaction
* Sends requests to FastAPI backend
* Displays sentiment result in real-time

---

## 🗂 Project Folder Structure

```text
TwitterSentimentVSCode/
│
├── src/
│   ├── app.py              # FastAPI backend
│   ├── preprocess.py       # Text preprocessing
│   ├── train_models.py     # Model training logic
│   ├── predict.py          # Prediction helper
│
├── models/                 # Model folder (ignored in Git)
│
├── frontend/
│   └── streamlit_app.py    # Streamlit UI
│
├── .github/
│   └── workflows/
│       └── docker-ci.yml   # CI/CD pipeline
│
├── Dockerfile
├── requirements.txt
├── .gitignore
├── README.md
└── main.py
```

---

## ▶️ How to Run Locally

### 1️⃣ Clone Repository

```bash
git clone https://github.com/khanarhamali/Twitter-Sentiment-Analysis-App.git
cd Twitter-Sentiment-Analysis-App
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run FastAPI Backend

```bash
uvicorn src.app:app --reload
```

Swagger UI:

```
http://127.0.0.1:8000/docs
```

### 5️⃣ Run Streamlit Frontend

```bash
streamlit run frontend/streamlit_app.py
```

---

## 🐳 Docker Support

### Build Docker Image

```bash
docker build -t khanarhamali/twitter-sentiment-app .
```

### Run Docker Container

```bash
docker run -p 8000:8000 khanarhamali/twitter-sentiment-app
```

---

## 🔁 CI/CD Pipeline (GitHub Actions)

This project includes a **CI/CD pipeline** that:

* Triggers on push to `main`
* Builds Docker image
* Logs in to Docker Hub
* Pushes image automatically

### 📁 Workflow Location

```
.github/workflows/docker-ci.yml
```

---

## 🔐 Environment Variables

| Variable    | Description                           |
| ----------- | ------------------------------------- |
| `MODEL_URL` | Public URL of trained model (Dropbox) |
| `PORT`      | Auto-set by Render                    |

---

## ☁️ Deployment

* Backend deployed on **Render**
* Model downloaded at runtime from **Dropbox**
* No large files stored in GitHub

---

## 🛠 Tools & Technologies

* Python
* Scikit-learn
* FastAPI
* Streamlit
* MLflow
* Docker
* GitHub Actions (CI/CD)
* Render
* Dropbox (model hosting)
* VS Code

---

## 👨‍💻 Author

**Arham Ali Khan**
Machine Learning & Data Science Enthusiast
