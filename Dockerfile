FROM python:3.11-slim as builder

WORKDIR /app

COPY ./search_api/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

#ENV SENTENCE_TRANSFORMERS_HOME=/app/models
#RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
# no need to download the model for the BM25 retriever

FROM python:3.11-slim as final
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

#ENV SENTENCE_TRANSFORMERS_HOME=/app/models
#COPY --from=builder /app/models /app/models

COPY . .

EXPOSE 8000

CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "search_api.main:app"]
