from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class Settings(BaseSettings):
    # Model configuration to load from .env files
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # OIDC Authentication
    OIDC_ISSUER_URL: str = os.getenv(
        "OIDC_ISSUER_URL", "https://datagems-dev.scayle.es/oauth/realms/dev"
    )
    OIDC_AUDIENCE: str = os.getenv("OIDC_AUDIENCE", "cross-dataset-discovery-api")
    GATEWAY_API_URL: str = os.getenv(
        "GATEWAY_API_URL", "https://datagems-dev.scayle.es"
    )

    @property
    def OIDC_CONFIG_URL(self) -> str:
        return f"{self.OIDC_ISSUER_URL}/.well-known/openid-configuration"

    # Database
    DB_CONNECTION_STRING: str
    TABLE_NAME: str = "your_table_name"  # Provide a default or load from env

    # Application & Search Index
    ROOT_PATH: str = ""
    INDEX_PATH: str = "./search_index"  # Path for the component's index artifacts


settings = Settings()
