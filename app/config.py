from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str
    # Keep google_api_key for embeddings only (if needed)
    google_api_key: str = ""

    qdrant_url: str
    qdrant_api_key: str

    qdrant_collection_name: str = "hackathon"
    max_retries: int = 2
    relevance_threshold: float = 0.7

    class Config:
        env_file = ".env"


settings = Settings()