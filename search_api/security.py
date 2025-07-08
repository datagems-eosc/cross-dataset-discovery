import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
import structlog
from . import auth_config
from typing import List

logger = structlog.get_logger(__name__)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
_oidc_config = None
_jwks_keys = None
async def get_oidc_config():
    """Fetches and caches the OIDC discovery document."""
    global _oidc_config
    if _oidc_config is None:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(auth_config.OIDC_CONFIG_URL)
                response.raise_for_status()
                _oidc_config = response.json()
        except httpx.RequestError as e:
            logger.error("Failed to fetch OIDC configuration", url=auth_config.OIDC_CONFIG_URL, error=str(e))
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Authentication service is unavailable.")
    return _oidc_config
async def get_jwks_keys():
    """Fetches and caches the JSON Web Key Set (JWKS) containing public keys."""
    global _jwks_keys
    if _jwks_keys is None:
        oidc_config = await get_oidc_config()
        jwks_uri = oidc_config.get("jwks_uri")
        if not jwks_uri:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="jwks_uri not found in OIDC config.")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(jwks_uri)
                response.raise_for_status()
                _jwks_keys = response.json()
        except httpx.RequestError as e:
            logger.error("Failed to fetch JWKS keys", url=jwks_uri, error=str(e))
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Could not fetch public keys for token validation.")
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
            audience=auth_config.OIDC_AUDIENCE,
            issuer=auth_config.OIDC_ISSUER_URL
        )
        return payload
    except JWTError as e:
        logger.warning("JWT validation failed", error=str(e))
        raise credentials_exception
    except Exception as e:
        logger.error("An unexpected error occurred during token validation", error=str(e))
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
            logger.warning(
                "Authorization failed: User missing any of the required roles",
                required_roles=required_roles,
                user_subject=claims.get("sub"),
                user_roles=list(user_roles) 
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User does not have any of the required roles: {', '.join(required_roles)}."
            )
        
        # If the check passes, return the claims for use in the endpoint
        return claims
    return role_checker