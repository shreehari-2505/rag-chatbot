from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    groq_api_key: str
    pinecone_api_key: str
    pinecone_env: str = "us-east-1"
    index_prefix: str = "rag-doc-"
    uploads_dir: str = "uploads"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }

settings = Settings()