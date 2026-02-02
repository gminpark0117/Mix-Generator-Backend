from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    ENV: str = "dev"
    DATABASE_URL_ASYNC: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/atomix"
    DATABASE_URL_SYNC: str = "postgresql://postgres:postgres@localhost:5432/atomix"
    LOOKAHEAD_MS: int = 30000

    STORAGE_DIR: str = "storage"          # folder on disk
    STORAGE_BASE_URL: str = "/storage"    # URL prefix to serve files from


settings = Settings()
