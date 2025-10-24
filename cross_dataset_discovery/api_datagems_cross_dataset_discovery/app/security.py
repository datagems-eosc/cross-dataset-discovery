from typing import List, Set

import httpx
import structlog
from api_datagems_cross_dataset_discovery.app.config import settings
from api_datagems_cross_dataset_discovery.app.exceptions import (
    FailedDependencyException,
)
from api_datagems_cross_dataset_discovery.app.logging_config import (
    get_correlation_id,
)
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

logger = structlog.get_logger(__name__)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
_oidc_config = None
_jwks_keys = None


async def get_oidc_config():
    global _oidc_config
    if _oidc_config is None:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(settings.OIDC_CONFIG_URL)
                response.raise_for_status()
                _oidc_config = response.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                "Failed to fetch OIDC configuration due to HTTP error",
                url=str(e.request.url),
                status_code=e.response.status_code,
                response=e.response.text,
            )
            try:
                payload = e.response.json()
            except Exception:
                payload = e.response.text
            raise FailedDependencyException(
                source="OIDCProvider",
                status_code=e.response.status_code,
                correlation_id=get_correlation_id(),
                payload=payload,
                detail="Authentication service returned an error.",
            )
        except httpx.RequestError as e:
            logger.error(
                "Failed to fetch OIDC configuration due to network error",
                url=settings.OIDC_CONFIG_URL,
                error=str(e),
            )
            raise FailedDependencyException(
                source="OIDCProvider",
                status_code=503,
                correlation_id=get_correlation_id(),
                payload={"error": f"Network error: {type(e).__name__}"},
                detail="Authentication service is unavailable.",
            )
    return _oidc_config


async def get_jwks_keys():
    """Fetches and caches the JSON Web Key Set (JWKS) containing public keys."""
    global _jwks_keys
    if _jwks_keys is None:
        oidc_config = await get_oidc_config()
        jwks_uri = oidc_config.get("jwks_uri")
        if not jwks_uri:
            raise FailedDependencyException(
                source="OIDCProvider",
                status_code=500,
                detail="jwks_uri not found in OIDC config.",
                correlation_id=get_correlation_id(),
            )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(jwks_uri)
                response.raise_for_status()
                _jwks_keys = response.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                "Failed to fetch JWKS keys due to HTTP error",
                url=str(e.request.url),
                status_code=e.response.status_code,
                response=e.response.text,
            )
            try:
                payload = e.response.json()
            except Exception:
                payload = e.response.text
            raise FailedDependencyException(
                source="OIDCProvider",
                status_code=e.response.status_code,
                correlation_id=get_correlation_id(),
                payload=payload,
                detail="Could not fetch public keys for token validation.",
            )
        except httpx.RequestError as e:
            logger.error(
                "Failed to fetch JWKS keys due to network error",
                url=jwks_uri,
                error=str(e),
            )
            raise FailedDependencyException(
                source="OIDCProvider",
                status_code=503,
                correlation_id=get_correlation_id(),
                payload={"error": f"Network error: {type(e).__name__}"},
                detail="Could not fetch public keys for token validation.",
            )
    return _jwks_keys


async def get_current_user_claims(token: str = Depends(oauth2_scheme)) -> dict:
    """
    A FastAPI dependency that validates the JWT and returns its claims.
    This will be applied to protected endpoints.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        keys = await get_jwks_keys()
        payload = jwt.decode(
            token,
            keys,
            algorithms=["RS256"],
            audience=settings.OIDC_AUDIENCE,
            issuer=settings.OIDC_ISSUER_URL,
        )
        return payload
    except JWTError as e:
        logger.warning("JWT validation failed", error=str(e))
        raise credentials_exception
    except Exception as e:
        logger.error(
            "An unexpected error occurred during token validation", error=str(e)
        )
        raise credentials_exception


def require_role(required_roles: List[str]):
    """
    A FastAPI dependency that checks if the user has at least one of the required roles.
    """

    def role_checker(claims: dict = Depends(get_current_user_claims)) -> dict:
        # this implementation  aims to check not only for user but for dg_user as well for the swagger
        user_roles = set(claims.get("realm_access", {}).get("roles", []))

        # Check for any intersection between the user's roles and the required roles
        if not user_roles.intersection(required_roles):
            log_context = {
                "required_roles": required_roles,
                "UserId": claims.get("sub"),
                "user_roles": list(user_roles),
            }
            client_id = claims.get("clientid")
            if client_id:
                log_context["ClientId"] = client_id

            logger.warning(
                "Authorization failed: User missing any of the required roles",
                **log_context,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User does not have any of the required roles: {', '.join(required_roles)}.",
            )

        # If the check passes, return the claims for use in the endpoint
        return claims

    return role_checker


async def get_authorized_dataset_ids(token: str) -> Set[str]:
    """
    Calls the DataGEMS Gateway to get the dataset IDs the user can access.
    """
    api_url = f"{settings.GATEWAY_API_URL}/api/principal/me/context-grants/dataset"
    headers = {"Authorization": f"Bearer {token}"}

    log = logger.bind(gateway_url=api_url)

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(api_url, headers=headers)
            response.raise_for_status()

            context_grants = response.json()
            # The API returns a list of grant objects. We need the 'targetId' from each.
            dataset_ids = {
                grant["targetId"] for grant in context_grants if "targetId" in grant
            }

            log.info(
                "Successfully fetched user dataset permissions.", count=len(dataset_ids)
            )
            return dataset_ids

    except httpx.HTTPStatusError as e:
        log.error(
            "Gateway returned an error when fetching dataset permissions.",
            status_code=e.response.status_code,
            response=e.response.text,
        )
        # Return an empty set on failure to prevent exposing data.
        return set()
    except Exception as e:
        log.error(
            "An unexpected error occurred while fetching dataset permissions.",
            error=str(e),
        )
        return set()
