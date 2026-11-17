FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

COPY web/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the quantized ONNX MiniLM model (~120 MB, no PyTorch)
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_dir='/app/.cache/fastembed', threads=1)"

COPY web/ .

EXPOSE 8081

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8081} --workers 1 --threads 4 --timeout 120 app:app"]
