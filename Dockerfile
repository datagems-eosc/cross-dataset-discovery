FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates && rm -rf /var/lib/apt/lists/*
ARG JDK_VERSION=21.0.4
ARG JDK_BUILD=7
ARG JDK_URL=https://github.com/adoptium/temurin21-binaries/releases/download/jdk-${JDK_VERSION}%2B${JDK_BUILD}/OpenJDK21U-jre_x64_linux_hotspot_${JDK_VERSION}_${JDK_BUILD}.tar.gz
ARG JAVA_INSTALL_DIR=/opt/java/openjdk
RUN wget -O /tmp/openjdk.tar.gz ${JDK_URL} && \
    mkdir -p ${JAVA_INSTALL_DIR} && \
    tar -xzf /tmp/openjdk.tar.gz -C ${JAVA_INSTALL_DIR} --strip-components=1 && \
    rm /tmp/openjdk.tar.gz
ENV JAVA_HOME=${JAVA_INSTALL_DIR}
ENV PATH="${JAVA_HOME}/bin:${PATH}"
WORKDIR /app
COPY ./search_api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
FROM python:3.11-slim AS final

WORKDIR /app
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd -r appuser && useradd -r -g appuser appuser

ARG JAVA_INSTALL_DIR=/opt/java/openjdk
COPY --from=builder ${JAVA_INSTALL_DIR} ${JAVA_INSTALL_DIR}
ENV JAVA_HOME=${JAVA_INSTALL_DIR}
ENV PATH="${JAVA_HOME}/bin:${PATH}"
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/
COPY . .

RUN chown -R appuser:appuser /app
USER appuser


EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD wget -q -O - http://localhost:8000/health || exit 1
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "search_api.main:app"]
