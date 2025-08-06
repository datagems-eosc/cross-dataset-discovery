import os

OIDC_ISSUER_URL = os.getenv(
    "OIDC_ISSUER_URL", "https://datagems-dev.scayle.es/oauth/realms/dev"
)
OIDC_AUDIENCE = os.getenv("OIDC_AUDIENCE", "cross-dataset-discovery-api")
OIDC_CONFIG_URL = f"{OIDC_ISSUER_URL}/.well-known/openid-configuration"
