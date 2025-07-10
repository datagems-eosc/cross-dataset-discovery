FROM python:3.11-slim as builder

WORKDIR /app
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    echo "deb http://deb.debian.org/debian bookworm-backports main" > /etc/apt/sources.list.d/backports.list && \
    apt-get update && \
    apt-get install -y -t bookworm-backports openjdk-21-jre-headless && \
    rm -rf /var/lib/apt/lists/*

COPY ./search_api/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt
FROM python:3.11-slim as final
WORKDIR /app
COPY --from=builder /usr/lib/jvm/ /usr/lib/jvm/
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

COPY . .

EXPOSE 8000

CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "search_api.main:app"]