FROM condaforge/mambaforge:latest as builder
WORKDIR /app
COPY ./search_api/environment.yml .
COPY ./search_api/requirements.txt .
RUN mamba env create -f environment.yml
RUN conda run -n search_env pip install --no-cache-dir -r requirements.txt
FROM condaforge/mambaforge:latest as final
WORKDIR /app
COPY --from=builder /opt/conda/envs/search_env /opt/conda/envs/search_env
COPY ./pyserini_collections ./pyserini_collections
COPY . .
ENV PATH /opt/conda/envs/cros_dataset_discovery_env/bin:$PATH
ENV PYSERINI_HOME=/app/pyserini_collections
EXPOSE 8000
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "search_api.main:app"]