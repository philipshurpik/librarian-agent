from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App configuration, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    sqlite_path: str = 'data/library.db'
    catalog_path: str = 'data/catalog.json'
    chunk_max_chars: int = 1500


settings = Settings()
