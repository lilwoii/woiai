# backend/config.py
import os
from pydantic import BaseSettings


class Settings(BaseSettings):
    # Broker / data keys (add/change as needed)
    ALPACA_API_KEY: str | None = None
    ALPACA_API_SECRET: str | None = None
    ALPACA_BASE_URL: str | None = None

    BINANCE_API_KEY: str | None = None
    BINANCE_API_SECRET: str | None = None

    # Discord
    DISCORD_WEBHOOK_URL: https://discord.com/api/webhooks/1430822328681631856/I17VeKRVIr3atYbBVS61pW4vLRUCtgQ87V8JmNcmLAFwy-QpBv7sCBTjsNT1BPEuNTNf

    # Heavy AI model
    HEAVY_MODEL_PATH: str = "models/heavy_trend_model.pt"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
