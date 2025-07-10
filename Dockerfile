FROM mambaorg/micromamba:1.5.8-bookworm as builder
WORKDIR /app
COPY ./search_api/environment.yml ./environment.yml
COPY ./search_api/requirements.txt ./requirements.txt
RUN micromamba create -n cross-dataset-discovery-env -f environment.yml --yes
RUN micromamba run -n cross-dataset-discovery-env pip install -r requirements.txt
FROM mambaorg/micromamba:1.5.8-bookworm
ARG MAMBA_ENV_NAME=cross-dataset-discovery-env
COPY --from=builder /opt/conda/envs/${MAMBA_ENV_NAME} /opt/conda/envs/${MAMBA_ENV_NAME}
COPY . .
WORKDIR /app
ENTRYPOINT ["micromamba", "run", "-n", "cross-dataset-discovery-env", "--"]
EXPOSE 8000
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "search_api.main:app"]