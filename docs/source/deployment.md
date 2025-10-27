# Deployment

The Cross-Dataset Discovery service is designed for containerized deployment using Docker. This guide provides instructions for building the image, configuring the service, and understanding its dependencies.

## Docker

The primary method for deploying the service is via a Docker container. The repository includes a `Dockerfile` for the build process.

### Dockerfile Stages

1.  **Builder Stage:**
    - Starts from a `python:3.11-slim` base image.
    - Installs OpenJDK, which is a required dependency for the underlying search library (Pyserini/Lucene).
    - Installs all Python dependencies from `requirements.txt`.

2.  **Final Stage:**
    - Starts from a fresh `python:3.11-slim` image.
    - Creates a non-root user (`appuser`) for security.
    - Copies the installed OpenJDK and Python packages from the builder stage.
    - Copies the application source code into the container.
    - Sets `appuser` as the active user.
    - Exposes port `8000`.
    - Includes a `HEALTHCHECK` instruction that queries the `/health` endpoint to monitor container health.
    - The default command (`CMD`) starts the application using `gunicorn` with a `uvicorn` worker, making it production-ready.

### Build and Run

To build and run the container, use the standard Docker commands:

```bash
# 1. Build the Docker image
docker build -t cross-dataset-discovery .

# 2. Run the container
# Note: You must provide a volume mount for the search index and all required environment variables.
docker run -p 8000:8000 \
  -v /path/to/your/search_index:/app/search_index \
  -e DB_CONNECTION_STRING="your-db-connection-string" \
  -e OIDC_ISSUER_URL="your-oidc-issuer-url" \
  -e INDEX_PATH="/app/search_index" \
  --name cdd-service \
  cross-dataset-discovery
```

## Configuration

The service is configured entirely through environment variables. This allows for flexible deployment across different environments without changing the container image.

| Variable | Description | Example |
|----------|-------------|---------|
| OIDC_ISSUER_URL | The base URL of the OIDC identity provider for token validation. | https://datagems-dev.scayle.es/oauth/realms/dev |
| OIDC_AUDIENCE | The audience claim that must be present in the JWT. | cross-dataset-discovery-api |
| GATEWAY_API_URL | The base URL of the DataGEMS Gateway API, used to fetch user permissions. | https://datagems-dev.scayle.es/gw |
| DB_CONNECTION_STRING | The full connection string for the PostgreSQL database (taken as secret from Vault). | postgresql://user:pass@host:port/dbname |
| TABLE_NAME | The name of the table to check in the database during health checks. | your_table_name |
| IdpClientSecret | The client secret for the identity provider (taken as secret from Vault). | your-secret |
| ROOT_PATH | The path prefix for the API if it's running behind a reverse proxy. | /cdd |
| INDEX_PATH | The path inside the container to the directory containing the search index files. | /app/search_index |

## Dependencies

The service requires several external systems and resources to be available at runtime to function correctly.

### Runtime Dependencies

**Search Index:**
- **Description:** A pre-built Pyserini/Lucene index is required for the core search functionality. The service does not create this index; it only reads from it.
- **Requirement:** The index directory must be mounted into the container at the path specified by the INDEX_PATH environment variable.

**OIDC Provider:**
- **Description:** An OpenID Connect compliant identity provider (e.g., Keycloak) is necessary for authenticating users by validating their JWTs.
- **Requirement:** The OIDC_ISSUER_URL and OIDC_AUDIENCE must be correctly configured to point to the identity provider.

**DataGEMS Gateway:**
- **Description:** The service communicates with the DataGEMS Gateway API to fetch dataset-level permissions for users. This is needed for enforcing data access policies.
- **Requirement:** The GATEWAY_API_URL must be configured, and the service must have network access to it.

### Build-time Dependencies

**OpenJDK:**
- **Description:** The Java Development Kit is required by the Pyserini library, which is a Python wrapper around the Java-based Lucene search engine.
- **Requirement:** It is automatically installed during the Docker build process as defined in the Dockerfile.
