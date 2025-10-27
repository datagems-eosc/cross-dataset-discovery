# Configuration

The Cross-Dataset Discovery service is configured using a combination of a local configuration file and environment variables.

## Configuration files

For local development, the service can be configured using a `.env` file placed in the root of the repository. The application will automatically load variables from this file on startup.

> **Warning**
> The `.env` file is intended for **local development only** and should be added to your `.gitignore` file to prevent committing secrets to version control.

### Example `.env` file

```env
# .env

# OIDC Authentication
OIDC_ISSUER_URL="https://datagems-dev.scayle.es/oauth/realms/dev"
OIDC_AUDIENCE="cross-dataset-discovery-api"
GATEWAY_API_URL="https://datagems-dev.scayle.es/gw"

# Database
DB_CONNECTION_STRING="postgresql://user:password@localhost:5432/datagems_db"
TABLE_NAME="your_table_name"

# Application & Search Index
ROOT_PATH=""
INDEX_PATH="./search_index"

# Secrets (for local use only)
IdpClientSecret="your-local-dev-secret"
```

## Environment Overrides

In any environment, especially in production, environment variables will always override values set in the `.env` file. This is the standard and recommended way to configure the service when deploying it as a container.

The following table lists all available configuration variables:

| Variable | Description | Example |
|----------|-------------|---------|
| OIDC_ISSUER_URL | The base URL of the OIDC identity provider for token validation. | https://datagems-dev.scayle.es/oauth/realms/dev |
| OIDC_AUDIENCE | The audience claim that must be present in the JWT. | cross-dataset-discovery-api |
| GATEWAY_API_URL | The base URL of the DataGEMS Gateway API, used to fetch user permissions. | https://datagems-dev.scayle.es/gw |
| DB_CONNECTION_STRING | The full connection string for the PostgreSQL database. | postgresql://user:pass@host:port/dbname |
| TABLE_NAME | The name of the table to check in the database during health checks. | your_table_name |
| IdpClientSecret | The client secret for the identity provider. | your-secret |
| ROOT_PATH | The path prefix for the API if it's running behind a reverse proxy. | /cdd |
| INDEX_PATH | The path inside the container to the directory containing the search index files. | /app/search_index |

## Secrets

Certain configuration variables contain sensitive information and must be handled securely.

### Identified Secrets

* **DB_CONNECTION_STRING**: Contains the database username and password.
* **IdpClientSecret**: The client secret used for communication with the identity provider.
