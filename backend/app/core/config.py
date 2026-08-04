import os
import torch
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Union

class Settings(BaseSettings):
    PROJECT_NAME: str = "GeoSim API"
    API_V1_STR: str = "/api/v1"
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # Model Paths
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/best_model.pth")
    
    # Device setup: CUDA or CPU
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env"
    )

settings = Settings()
