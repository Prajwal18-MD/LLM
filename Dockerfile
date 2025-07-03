# Dockerfile

FROM python:3.12.5-slim

WORKDIR /app

# 1. Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 2. Download spaCy English model
RUN python -m spacy download en_core_web_sm

# 3. Copy app code
COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
