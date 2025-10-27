# API Overview

The Cross-Dataset Discovery service exposes a RESTful API for performing search operations.

## Base URL

The API is served from the root of the application. All endpoint paths are relative to the base URL where the service is deployed:

**Base URL:** [https://datagems-dev.scayle.es/cross-dataset-discovery/](https://datagems-dev.scayle.es/cross-dataset-discovery/)

## Authentication

All endpoints are protected and require a valid JWT Bearer token. See the [Security](./security.md) page for more details.

---

## Endpoints

### Search

Performs a search query against the indexed datasets.

*   **Endpoint:** `POST /search/`
*   **Description:** Submits a natural language query and returns a ranked list of relevant results. The search can be optionally filtered to a specific set of datasets. The results returned are automatically filtered based on the user's access permissions.
*   **Request Body:**

    ```json
    {
      "query": "string",
      "k": 5,
      "dataset_ids": [
        "string"
      ]
    }
    ```

*   **Response Body (Success):**

    ```json
    {
      "query_time": "retrieval time in seconds",
      "results": [
        {
          "content": "The main text content of the search result.",
          "dataset_id": "uuid-of-the-source-dataset",
          "object_id": "uuid-of-the-source-object",
          "similarity": "simialrity-score (higher the better)"
        }
      ]
    }
    ```

### Health Check

Verifies the operational status of the API and its dependencies.

*   **Endpoint:** `GET /health`
*   **Description:** Checks the availability of the search component and the existance of the index. Returns a `200 OK` if all systems are healthy.
*   **Response Body (Success):**

    ```json
    {
      "status": "ok",
      "message": "All dependencies are healthy."
    }
    ```
