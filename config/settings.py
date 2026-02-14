"""
Centralized configuration management for Properchy.

This module provides configuration settings for the ML pipeline,
including paths, model parameters, and feature engineering options.
"""

import os
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class PathConfig(BaseModel):
    """Configuration for file paths."""
    
    # Project root directory
    ROOT_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    
    # Data directories
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    EXTRACTED_DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "extracted_data")
    
    # MLflow directories
    MLRUNS_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "mlruns")
    
    # Data file
    DATA_FILE: str = "archive.zip"
    
    @property
    def data_file_path(self) -> Path:
        """Get full path to data file."""
        return self.DATA_DIR / self.DATA_FILE
    
    @field_validator("DATA_DIR", "EXTRACTED_DATA_DIR", "MLRUNS_DIR")
    @classmethod
    def create_directory(cls, v: Path) -> Path:
        """Create directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v


class ModelConfig(BaseModel):
    """Configuration for model training."""
    
    # Model type
    MODEL_TYPE: str = Field(default="xgboost", description="Model type: xgboost or linear_regression")
    
    # XGBoost parameters
    XGBOOST_PARAMS: dict = Field(
        default_factory=lambda: {
            "objective": "reg:squarederror",
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
        }
    )
    
    # Linear Regression parameters
    LINEAR_REGRESSION_PARAMS: dict = Field(
        default_factory=lambda: {
            "fit_intercept": True,
            "copy_X": True,
        }
    )
    
    # Random state for reproducibility
    RANDOM_STATE: int = 42
    
    # Test size for train-test split
    TEST_SIZE: float = Field(default=0.2, ge=0.1, le=0.5)
    
    @field_validator("MODEL_TYPE")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        """Validate model type."""
        allowed_types = ["xgboost", "linear_regression"]
        if v not in allowed_types:
            raise ValueError(f"Model type must be one of {allowed_types}")
        return v


class FeatureEngineeringConfig(BaseModel):
    """Configuration for feature engineering."""
    
    # Target column
    TARGET_COLUMN: str = "SalePrice"
    
    # Columns for log transformation
    LOG_TRANSFORM_FEATURES: List[str] = Field(
        default_factory=lambda: ["Gr Liv Area", "SalePrice"]
    )
    
    # Outlier detection column
    OUTLIER_DETECTION_COLUMN: str = "SalePrice"
    
    # Outlier detection method
    OUTLIER_METHOD: str = Field(default="iqr", description="Method: iqr or zscore")
    
    # IQR multiplier for outlier detection
    IQR_MULTIPLIER: float = Field(default=1.5, ge=1.0, le=3.0)
    
    # Z-score threshold for outlier detection
    ZSCORE_THRESHOLD: float = Field(default=3.0, ge=2.0, le=5.0)
    
    @field_validator("OUTLIER_METHOD")
    @classmethod
    def validate_outlier_method(cls, v: str) -> str:
        """Validate outlier detection method."""
        allowed_methods = ["iqr", "zscore"]
        if v not in allowed_methods:
            raise ValueError(f"Outlier method must be one of {allowed_methods}")
        return v


class MLflowConfig(BaseModel):
    """Configuration for MLflow tracking."""
    
    # Experiment name
    EXPERIMENT_NAME: str = "my_experiment_name"
    
    # Tracking URI
    TRACKING_URI: str = "file:./mlruns"
    
    # Model registry URI (optional)
    REGISTRY_URI: Optional[str] = None
    
    # Enable autologging
    AUTOLOG_ENABLED: bool = True
    
    # Log model artifacts
    LOG_MODELS: bool = True


class ZenMLConfig(BaseModel):
    """Configuration for ZenML."""
    
    # Stack name
    STACK_NAME: str = "local_mlflow_stack"
    
    # Experiment tracker name
    EXPERIMENT_TRACKER_NAME: str = "mlflow_tracker"
    
    # Model deployer name
    MODEL_DEPLOYER_NAME: str = "mlflow_deployer"
    
    # Enable caching
    ENABLE_CACHE: bool = False


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    
    # Log level
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # Log format
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Log to file
    LOG_TO_FILE: bool = False
    
    # Log file path
    LOG_FILE: Optional[Path] = None
    
    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in allowed_levels:
            raise ValueError(f"Log level must be one of {allowed_levels}")
        return v_upper


class Settings(BaseModel):
    """Main settings class combining all configurations."""
    
    paths: PathConfig = Field(default_factory=PathConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    features: FeatureEngineeringConfig = Field(default_factory=FeatureEngineeringConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    zenml: ZenMLConfig = Field(default_factory=ZenMLConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    class Config:
        """Pydantic config."""
        validate_assignment = True
        extra = "forbid"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get global settings instance."""
    return settings


def update_settings(**kwargs) -> Settings:
    """Update settings with new values."""
    global settings
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    return settings


if __name__ == "__main__":
    # Example usage
    config = get_settings()
    print(f"Data path: {config.paths.data_file_path}")
    print(f"Model type: {config.model.MODEL_TYPE}")
    print(f"Target column: {config.features.TARGET_COLUMN}")
