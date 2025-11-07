# Security

## Authentication

Authentication is handled via the **OAuth 2.0** and **OpenID Connect (OIDC)** protocols. Every request to a protected endpoint must include a valid JSON Web Token (JWT) in the `Authorization` header as a Bearer token.

The service validates incoming JWTs against the OIDC provider's public keys. The token's signature, issuer, and audience are all verified to ensure its authenticity.

## Authorization

Authorization is role-based. The service checks for specific roles within the validated JWT's claims.

- **Required Roles:** To access the `/search/` endpoint, the user's token must contain at least one of the following roles in its `realm_access.roles` claim:
    - `user`
    - `dg_user`

If a user attempts to access an endpoint without the required role, the API will respond with a `403 Forbidden` error.

## Dataset-Level Permissions

In addition to role-based access, the service enforces dataset-level permissions.

Before executing a search, the API performs a token exchange and then calls the DataGEMS Gateway's `POST /api/principal/context-grants/query` endpoint. It requests all grants associated with roles like `dg_ds-browse` to get a list of all dataset IDs the user is permitted to access.

- **Search Filtering:**
    - If a user's search request includes `dataset_ids`, the service intersects this list with the user's authorized datasets. The search is only performed on the datasets present in both lists.
    - All search results, regardless of the input filter, are *always* filtered against the user's authorized dataset list before being returned. This ensures a user can never see data from a dataset they are not permitted to access.
