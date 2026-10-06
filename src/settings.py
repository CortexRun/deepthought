from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables or a local .env file.

    See .env.example for the full list. Database credentials can come either
    from PG_* variables or from a libpq-style .pgpass file (PGPASS_PATH /
    PGPASS_INDEX); the .pgpass entry wins if both are present.
    """

    app_name: str = "Deepthought API"
    host: str = "0.0.0.0"
    port: int = 8000

    # Ollama
    ollama_base_url: str = "http://localhost:11434"

    # PostgreSQL / pgvector
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_database: str = "deepthought"
    pg_user: str = "deepthought"
    pg_password: str = ""

    # Optional: read credentials from a .pgpass file instead
    pgpass_path: Optional[str] = None
    pgpass_index: int = 0

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
