import os
from functools import lru_cache

from pydantic.v1 import BaseSettings



class Settings(BaseSettings):
    ENV: str = "development"
    MONGO_URI: str = "mongodb://localhost:27017"
    OPENAI_API_KEY: str = ""
    PINECONE_API_KEY: str = ""
    PINECONE_ENV: str = ""
    PINECONE_INDEX: str | None = None
    DEFAULT_PASS: str = "changeme!"
    SESSION_SECRET: str = "dev-secret"
    OWNER_USER: str = "owner"
    OWNER_PASS: str = "changeme!"

    class Config:

        env_file = ".env.production" if os.getenv("ENV", "development").lower() == "production" else ".env.development"

        case_sensitive = False

    @property
    def db_name(self) -> str:
        return "recruitment_app" if self.ENV == "production" else "recruitment_app_dev"

    @property
    def pinecone_index(self) -> str:
        if self.PINECONE_INDEX:
            return self.PINECONE_INDEX
        return "resumes-index" if self.ENV == "production" else "resumes-dev"


@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
