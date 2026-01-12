# Twitter Sentiment Analysis API 

This project is a **complete end-to-end Machine Learning + API deployment system** for **Twitter Sentiment Analysis**. It covers the full lifecycle starting from data preprocessing and model training to experiment tracking with MLflow and deployment using **FastAPI** on **Render**.

---

## Project Overview

The goal of this project is to classify tweet text into **Positive**, **Neutral**, or **Negative** sentiment using traditional Machine Learning models.

### Key Highlights

* Text preprocessing using **NLTK**
* Model training using **Scikit-learn**
* Handling class imbalance with **imbalanced-learn**
* Experiment tracking with **MLflow**
* REST API using **FastAPI**
* Cloud deployment on **Render**
* Model loading from **Dropbox** (no large files in Git)

---

## Machine Learning Models Used

The following models were trained and evaluated:

* **Logistic Regression**
* **Random Forest Classifier** ✅ (final selected model)

### Final Model

* **RandomForest_pipeline.pkl**
* Selected based on better overall performance (F1-score & accuracy)
* Saved as a **Scikit-learn Pipeline**

---

## Preprocessing Techniques

Text preprocessing is handled inside `src/preprocess.py` using **NLTK**.

### Steps:

* Lowercasing text
* Removing URLs
* Removing mentions (@user)
* Removing hashtags
* Removing punctuation & special characters
* Tokenization
* Stopword removal
* Lemmatization

These steps ensure clean and normalized input before prediction.

---

## Experiment Tracking (MLflow)

MLflow is used to:

* Track different model runs
* Log hyperparameters
* Log evaluation metrics
* Save trained models

### MLflow Components Used:

* **Experiments**
* **Runs**
* **Metrics** (accuracy, precision, recall, F1-score)
* **Artifacts** (trained models)

MLflow can be run locally using:

```bash
mlflow ui
```

---

## API Layer (FastAPI)

The trained model is exposed as a REST API using **FastAPI**.

### Endpoints

#### Root Endpoint

```http
GET /
```

Response:

```json
{
  "status": "Twitter Sentiment API is running"
}
```

#### Prediction Endpoint

```http
POST /predict
```

Request Body:

```json
{
  "text": "I love this game!"
}
```

Response:

```json
{
  "sentiment": 1,
  "label": "Positive"
}
```

### Sentiment Mapping

| Model Output | Sentiment |
| ------------ | --------- |
| -1           | Negative  |
| 0            | Neutral   |
| 1            | Positive  |

---

## 🛠️ Tools & Technologies Used

* **Python 3.10**
* **VS Code** (development environment)
* **Scikit-learn** (ML models & pipelines)
* **NLTK** (text preprocessing)
* **imbalanced-learn** (class imbalance handling)
* **MLflow** (experiment tracking)
* **FastAPI** (API development)
* **Uvicorn** (ASGI server)
* **Dropbox** (model hosting)
* **Render** (deployment)

---

## Project Structure

```
project-root/
│
├── src/
│   ├── app.py              # FastAPI app
│   ├── preprocess.py       # Text preprocessing
│   ├── train_models.py     # Model training
│   ├── predict.py          # Prediction logic
│
├── models/                 # Model downloaded at runtime
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python version (3.10)
├── .gitignore
└── README.md
```

---

## Installation & Setup (Local)

### 1️. Clone Repository

```bash
git clone <your-repo-url>
cd project-root
```

### 2️. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

### 3️. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️. Download NLTK Resources

```bash
python -m nltk.downloader punkt stopwords wordnet
```

---

##  Run the Project

### Run FastAPI App Locally

```bash
uvicorn src.app:app --reload
```

Open browser:

```
http://127.0.0.1:8000
```

Swagger Docs:

```
http://127.0.0.1:8000/docs
```

---

## Deployment (Render)

* Model file is **not pushed to GitHub**
* Model is downloaded automatically from **Dropbox** at runtime

### Start Command (Render)

```
uvicorn src.app:app --host 0.0.0.0 --port $PORT
```

### Build Command

```
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords wordnet
```

---

## ✅ Key Design Decisions

* ❌ No large `.pkl` files in GitHub
* ✅ Cloud-friendly model loading
* ✅ Reproducible ML experiments with MLflow
* ✅ Clean ML pipeline architecture
* ✅ Production-ready API

---

## Future Improvements

* Add Docker support
* Add CI/CD pipeline
* Add authentication to API
* Add batch prediction endpoint
* Add monitoring & logging

---

## Author

**Arham Ali Khan**
Machine Learning | Data Science | MLOps
