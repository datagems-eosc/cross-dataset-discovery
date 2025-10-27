# Datastores

The Cross-Dataset Discovery service relies on a search index for fast retrieval.

## Search Index

The core of the retrieval functionality is powered by pre-built search indexes. The API is configured to use an index compatible with **Pyserini (which is based on Apache Lucene)**.

- **Technology:** Apache Lucene
- **Purpose:** Provides fast, full-text search capabilities for the `/search/` endpoint.
- **Location:** The index files are stored on the local filesystem. The path to the index directory must be provided to the running container via a volume mount.
