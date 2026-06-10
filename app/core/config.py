"""Application configuration."""

from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # App settings
    APP_NAME: str = "Brecha AI Worker"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    PORT: int = 8081
    WORKER_TIMEOUT: int = 300  # 5 minutes timeout per message

    # GCP Pub/Sub settings
    GCP_PROJECT_ID: str
    PUBSUB_SUBSCRIPTION_ID: str = "brecha-worker-sub"
    PUBSUB_CATALOG_REFRESH_SUBSCRIPTION_ID: str = "brecha-catalog-refresh-sub"
    PUBSUB_MAX_MESSAGES: int = 1  # Process one message at a time
    PUBSUB_ACK_DEADLINE: int = 600  # 10 minutes to process message

    # Vertex AI / Gemini settings
    GCP_LOCATION: str = "global"
    GEMINI_MODEL_NAME: str = "gemini-3.1-pro-preview"
    GEMINI_MAX_RETRIES: int = 3
    GEMINI_RETRY_DELAY: int = 2

    # Supabase settings
    SUPABASE_URL: str
    SUPABASE_KEY: str
    SUPABASE_TABLE: str = "project_queries"
    SUPABASE_CLASSIFICATIONS_TABLE: str = "project_classifications"

    # BETO classifier
    BETO_MODEL_DIR: str = "modelo_beto"
    BETO_BASE_MODEL: str = "dccuchile/bert-base-spanish-wwm-cased"

    # Worker settings
    MAX_WORKERS: int = 1

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


# Create settings instance
settings = Settings()
