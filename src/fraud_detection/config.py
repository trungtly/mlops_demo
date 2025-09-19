"""Configuration management for fraud detection system."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseSettings


class Config(BaseSettings):
    """Base configuration class."""
    
    # Project structure
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"
    
    # Artifacts and experiments
    ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
    MODELS_DIR: Path = ARTIFACTS_DIR / "models"
    EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"
    MODEL_REGISTRY_DIR: Path = PROJECT_ROOT / "model_registry"
    
    # Logging
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    LOG_LEVEL: str = "INFO"
    
    # MLflow
    MLFLOW_TRACKING_URI: str = "sqlite:///experiments/mlflow.db"
    MLFLOW_EXPERIMENT_NAME: str = "fraud_detection"
    MLFLOW_ARTIFACT_ROOT: Optional[str] = None
    
    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    
    # Model settings
    MODEL_NAME: str = "fraud_detector"
    MODEL_VERSION: str = "latest"
    PREDICTION_THRESHOLD: float = 0.5
    
    # Data settings
    DATASET_URL: str = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
    TARGET_COLUMN: str = "Class"
    TIME_COLUMN: str = "Time"
    AMOUNT_COLUMN: str = "Amount"
    
    # Training settings
    RANDOM_SEED: int = 42
    TEST_SIZE: float = 0.2
    VAL_SIZE: float = 0.2
    
    # Monitoring
    DRIFT_THRESHOLD: float = 0.1
    PERFORMANCE_THRESHOLD: float = 0.8
    
    class Config:
        env_file = ".env"
        case_sensitive = True


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_config() -> Config:
    """Get application configuration."""
    return Config()


def create_directories(config: Config) -> None:
    """Create necessary directories."""
    directories = [
        config.DATA_DIR,
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.EXTERNAL_DATA_DIR,
        config.ARTIFACTS_DIR,
        config.MODELS_DIR,
        config.EXPERIMENTS_DIR,
        config.MODEL_REGISTRY_DIR,
        config.LOGS_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = get_config()
create_directories(config)