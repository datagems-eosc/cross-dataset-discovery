# System Architecture

The Cross-Dataset Discovery Service is a self-contained application designed to act as a specialized search layer within the DataGEMS ecosystem.

## Core Components

1.  **FastAPI Application:** The heart of the service is a Python web application built with FastAPI. It exposes the REST API, handles incoming HTTP requests, and orchestrates all internal operations.

2.  **Search Component:** An internal module responsible for executing queries against the search index. It takes the user query and returns a ranked list of document IDs.

3.  **Security Layer:** This layer intercepts all incoming requests to perform authentication and authorization. It integrates with an external OIDC provider to validate JWTs and with the DataGEMS Gateway to fetch user-specific permissions.

## Request Flow

A typical search request follows this path:

1.  A user sends a `POST /search/` request with a valid JWT.
2.  The Security Layer validates the token and checks for the required user roles.
3.  The service calls the DataGEMS Gateway to get the list of datasets the user is authorized to see.
4.  The user's query is passed to the Search Component, which queries the pre-built index returning an authorized dataset list.
5.  The final, authorized results are returned to the user.
