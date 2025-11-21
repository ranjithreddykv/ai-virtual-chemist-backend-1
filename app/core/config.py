# app/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Virtual Chemist"
    API_V1_STR: str = "/api/v1"
    BACKEND_CORS_ORIGINS: list[str] = ["*"]  # Update this in production
    DEBUG: bool = True

    # Model names (optional, for clarity)
    FORWARD_MODEL_NAME: str = "sagawa/ReactionT5v2-forward"
    RETRO_MODEL_NAME: str = "sagawa/ReactionT5v2-retro"
    YIELD_MODEL_NAME: str = "sagawa/ReactionT5v2-yield"

    class Config:
        env_file = ".env"
        case_sensitive = True


# Instantiate global settings object
settings = Settings()
