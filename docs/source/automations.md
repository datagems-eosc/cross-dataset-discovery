# Automations

This project leverages GitHub Actions to automate builds, code analysis, security scanning, and documentation deployment. The following sections detail the specific workflows configured for this repository.

## Dockerfile

The creation of a production-ready container image is fully automated by the `Dockerfile` in the root of the repository. This build process ensures a consistent environment for the application.

-   **Stage 1 (Builder):** This stage prepares the environment by installing all necessary build-time dependencies, including OpenJDK (for the search library) and the full set of Python packages.
-   **Stage 2 (Final):** This stage constructs the final, lean image by copying only the essential artifacts from the builder stage. It creates a non-root `appuser` for security, copies the application code, and sets the `gunicorn` server as the entry point.

## Docker image publishing

**Workflow:** `.github/workflows/docker-publish.yml`

This workflow automates the process of building and publishing the service's Docker image to the GitHub Container Registry (ghcr.io).

-   **Trigger:** This workflow runs automatically whenever a new Git tag matching the pattern `v*` (e.g., `v1.0.0`, `v1.1.0`) is pushed to the repository.
-   **Process:**
    1.  Logs into the GitHub Container Registry using a temporary `GITHUB_TOKEN`.
    2.  Uses the `docker/metadata-action` to automatically generate appropriate Docker tags based on the Git tag.
    3.  Builds the Docker image using the `Dockerfile`.
    4.  Pushes the newly built and tagged image to the `ghcr.io/datagems-eosc/cross-dataset-discovery-api` repository, making it available for deployment.

## Vulnerability Scanning

**Workflow:** `.github/workflows/vulnerability-scan-on-demand.yml`

This workflow provides on-demand security scanning of the project's configuration and Docker images using the **Trivy** security scanner.

-   **Trigger:** This workflow is run manually from the GitHub Actions UI (`workflow_dispatch`). It requires an `image_tag` as input to specify which published Docker image to scan.
-   **Process:**
    1.  **Configuration Scan:** Scans the `Dockerfile` and other repository configuration files for potential security misconfigurations.
    2.  **Image Scan:** Pulls the specified Docker image from the GitHub Container Registry and scans its operating system packages and Python libraries for known vulnerabilities (CVEs) with `CRITICAL` or `HIGH` severity.
    3.  **Artifacts:** The JSON reports from both the configuration scan and the image scan are uploaded as build artifacts for review.

## Static Code Analysis

**Workflow:** `.github/workflows/code-scan-on-demand.yml`

This workflow performs in-depth static code analysis using **GitHub CodeQL** to find potential bugs, security vulnerabilities, and quality issues in the codebase.

-   **Trigger:** This workflow is run manually from the GitHub Actions UI (`workflow_dispatch`).
-   **Process:**
    1.  It runs two separate analysis jobs: one for the Python source code and one for the GitHub Actions workflow files themselves.
    2.  For each language, it initializes CodeQL using an extended set of security and quality queries.
    3.  It performs the analysis and uploads the results directly to the repository's "Security" tab under "Code scanning alerts".

## Code Metrics

**Workflow:** `.github/workflows/code-metrics-on-demand.yml`

This workflow generates a report on the complexity and maintainability of the application's Python code using the **Radon** library.

-   **Trigger:** This workflow is run manually from the GitHub Actions UI (`workflow_dispatch`).
-   **Process:**
    1.  Installs the `radon` Python package.
    2.  Runs three types of analysis on the `cross_dataset_discovery/api_datagems_cross_dataset_discovery/app` directory:
        -   **Maintainability Index (MI):** A score from 0-100 indicating how easy the code is to maintain.
        -   **Cyclomatic Complexity (CC):** Measures the number of independent paths through the code, indicating its complexity.
        -   **Raw Lines of Code (LOC):** Provides basic code size metrics.
    3.  The complete report is compiled into a text file and uploaded as a build artifact named `code-metrics-report`.

## Documentation

**Workflow:** `.github/workflows/deploy-docs-on-demand.yml`

The deployment of this documentation site is automated by a dedicated GitHub Action workflow.

-   **Trigger:** This workflow is run manually from the GitHub Actions UI (`workflow_dispatch`).
-   **Process:** When triggered with a version number (e.g., `1.0.0`), the workflow:
    1.  Installs `mkdocs`, `mike`, and other required tools.
    2.  Builds the static HTML site from the markdown source files in the `docs/` directory.
    3.  Uses the `mike` tool to commit the built site to the `gh-pages` branch, organized under the specified version.
    4.  Updates the `latest` alias to point to this newly deployed version, ensuring visitors see the most recent documentation by default.
