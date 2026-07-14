from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App configuration, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    openai_api_key: str = ''
    embedding_model: str = 'text-embedding-3-small'
    qdrant_url: str = 'http://localhost:6333'
    qdrant_collection_prefix: str = 'books'
    sqlite_path: str = 'data/library.db'
    catalog_path: str = 'data/catalog.json'
    chunk_max_chars: int = 1500

    @property
    def qdrant_collection(self) -> str:
        """One collection per embedding model: a model switch lands in a fresh collection (blue/green)."""
        return f'{self.qdrant_collection_prefix}__{self.embedding_model.replace("text-embedding-", "openai-")}'


settings = Settings()
