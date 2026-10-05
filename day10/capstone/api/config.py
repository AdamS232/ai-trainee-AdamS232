"""App settings, loaded from day10/capstone/.env (with safe defaults)."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="day10/capstone/.env", extra="ignore")

    ollama_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chroma_dir: str = "day10/capstone/chroma"
    upload_dir: str = "day10/capstone/data/uploads"
    retrieve_k: int = 20            # candidates from vector search
    top_n: int = 4                  # chunks kept after re-ranking
    min_rerank_score: float = -6.0  # below this, answer "I don't know" without calling the LLM


settings = Settings()