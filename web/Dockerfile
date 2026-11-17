FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Install CPU-only PyTorch first to keep the image small and fast to build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY web/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the MiniLM embedding model so container startup is fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

COPY web/ .

EXPOSE 8081

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8081} --workers 1 --threads 4 --timeout 120 app:app"]
