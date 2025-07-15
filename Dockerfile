FROM python:3.11-slim AS builder
WORKDIR /app
ENV HF_HOME=/app/.cache
COPY ./search_api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
FROM python:3.11-slim AS final
WORKDIR /app
ENV HF_HOME=/app/.cache
COPY --from=builder /app/.cache /app/.cache
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/
COPY . .

EXPOSE 8000
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "search_api.main:app"]