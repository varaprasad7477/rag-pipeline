from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    data_dir: Path = Path("data")
    llm_provider: str = "extractive"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4.1-mini"
    llm_api_key: str = ""
    top_k: int = 5
    chunk_size: int = 900
    chunk_overlap: int = 140
    max_file_bytes: int = 10 * 1024 * 1024

    @property
    def database_path(self) -> Path:
        return self.data_dir / "rag.db"


settings = Settings()
