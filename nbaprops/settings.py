"""Environment settings (secrets, connection strings). Loaded from .env or the environment."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://nba:nba@localhost:5432/nbaprops"
    propline_api_key: str = ""
    odds_api_key: str = ""
    discord_webhook_url: str = ""
    local_tz: str = "America/New_York"
    strategy_config: str = "config/strategy.yaml"


@lru_cache
def get_settings() -> Settings:
    return Settings()
