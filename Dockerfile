# 1. Use a stable, small base
FROM python:3.12.5-slim-bullseye

WORKDIR /app

# 2. System updates and Python deps
RUN apt-get update \
 && apt-get upgrade -y \
 && pip install --upgrade pip \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && python -m spacy download en_core_web_sm

# 3. App code
COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
