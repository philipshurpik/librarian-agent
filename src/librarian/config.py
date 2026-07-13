from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App configuration, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    openai_api_key: str = ''
    embedding_model: str = 'text-embedding-3-small'
    sqlite_path: str = 'data/library.db'
    catalog_path: str = 'data/catalog.json'
    chunk_max_chars: int = 1500


settings = Settings()
