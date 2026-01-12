FROM python:3.10-slim

# Set working directory
WORKDIR /app

# System dependencies (for nltk, sklearn)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Download NLTK data
RUN python - <<EOF
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
EOF

# Copy project files
COPY . .

# Expose port (Render / Docker)
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
