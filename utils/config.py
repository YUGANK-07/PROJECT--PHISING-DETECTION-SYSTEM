"""
utils/config.py
───────────────
Centralised, type-safe settings loaded from environment variables
(and optionally from a .env file).  All other modules import from
here — never read os.environ directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration.

    Fields map 1-to-1 with .env.example.  pydantic-settings automatically
    reads from the .env file and environment variables, with env vars taking
    precedence.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    APP_ENV: Literal["development", "production", "test"] = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    APP_SECRET_KEY: str = "change-me-to-a-long-random-string"
    LOG_LEVEL: str = "INFO"

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "phishing_detector"
    POSTGRES_USER: str = "phishing_user"
    POSTGRES_PASSWORD: str = "change-me"

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # ── MongoDB ───────────────────────────────────────────────────────────────
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB: str = "phishing_html_cache"

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    REDIS_DB: int = 0
    REDIS_TTL_DNS: int = 3600
    REDIS_TTL_WHOIS: int = 86400
    REDIS_TTL_PREDICTION: int = 300

    @property
    def redis_url(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # ── Data Sources ──────────────────────────────────────────────────────────
    PHISHTANK_API_KEY: str = ""
    PHISHTANK_FEED_URL: str = (
        "http://data.phishtank.com/data/online-valid.json"
    )
    OPENPHISH_FEED_URL: str = "https://openphish.com/feed.txt"
    TRANCO_LIST_URL: str = "https://tranco-list.eu/top-1m.csv.zip"

    # ── Model ─────────────────────────────────────────────────────────────────
    MODEL_ARTIFACT_DIR: Path = Path("models/artifacts")
    BERT_MODEL_NAME: str = "distilbert-base-uncased"
    INFERENCE_DEVICE: Literal["cpu", "cuda"] = "cpu"

    # ── Auth / Security ───────────────────────────────────────────────────────
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60
    RATE_LIMIT_PER_MINUTE: int = 60
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_origins(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",")]
        return v

    # ── Derived Paths ─────────────────────────────────────────────────────────
    @property
    def raw_data_dir(self) -> Path:
        return Path("data/raw")

    @property
    def processed_data_dir(self) -> Path:
        return Path("data/processed")

    def model_post_init(self, __context) -> None:  # noqa: D401
        """Create required directories on startup."""
        for d in [
            self.raw_data_dir,
            self.processed_data_dir,
            self.MODEL_ARTIFACT_DIR,
        ]:
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()


# Convenient module-level alias
settings: Settings = get_settings()
