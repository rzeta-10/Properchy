"""
Model building strategies for house price prediction.

This module implements the Strategy pattern for different ML models,
providing a flexible and extensible approach to model training.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Setup logging
logger = logging.getLogger(__name__)


class ModelBuildingStrategy(ABC):
    """
    Abstract base class for model building strategies.
    
    This class defines the interface that all concrete model building
    strategies must implement.
    """
    
    @abstractmethod
    def build_and_train_model(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Pipeline:
        """
        Build and train a model.

        Args:
            X_train: Training data features
            y_train: Training data labels/target

        Returns:
            Trained scikit-learn pipeline

        Raises:
            TypeError: If inputs are not pandas DataFrame/Series
            ValueError: If data is invalid or empty
        """
        pass


class LinearRegressionStrategy(ModelBuildingStrategy):
    """
    Linear Regression model building strategy.
    
    This strategy builds a pipeline with standard scaling and
    linear regression for baseline predictions.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Linear Regression strategy.
        
        Args:
            params: Optional parameters for LinearRegression model
        """
        self.params = params or {"fit_intercept": True}
        logger.info(f"Initialized LinearRegressionStrategy with params: {self.params}")
    
    def build_and_train_model(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Pipeline:
        """
        Build and train a linear regression model.

        Args:
            X_train: Training data features
            y_train: Training data labels/target

        Returns:
            Trained pipeline with scaler and linear regression model

        Raises:
            TypeError: If inputs are not pandas DataFrame/Series
            ValueError: If data is empty or contains invalid values
        """
        # Validate inputs
        self._validate_inputs(X_train, y_train)
        
        logger.info("Building Linear Regression pipeline with scaling")
        
        try:
            # Create pipeline with standard scaling and linear regression
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LinearRegression(**self.params)),
            ])
            
            logger.info("Training Linear Regression model")
            pipeline.fit(X_train, y_train)
            
            logger.info("Model training completed successfully")
            return pipeline
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
    
    @staticmethod
    def _validate_inputs(X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Validate input data."""
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series")
        if X_train.empty or y_train.empty:
            raise ValueError("Training data cannot be empty")
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same length")


class XGBoostStrategy(ModelBuildingStrategy):
    """
    XGBoost model building strategy.
    
    This strategy builds a pipeline with standard scaling and
    XGBoost regressor for advanced gradient boosting predictions.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize XGBoost strategy.
        
        Args:
            params: Optional parameters for XGBRegressor model
        """
        default_params = {
            "objective": "reg:squarederror",
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
        }
        self.params = {**default_params, **(params or {})}
        logger.info(f"Initialized XGBoostStrategy with params: {self.params}")
    
    def build_and_train_model(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Pipeline:
        """
        Build and train an XGBoost model.

        Args:
            X_train: Training data features
            y_train: Training data labels/target

        Returns:
            Trained pipeline with scaler and XGBoost model

        Raises:
            TypeError: If inputs are not pandas DataFrame/Series
            ValueError: If data is empty or contains invalid values
        """
        # Validate inputs
        self._validate_inputs(X_train, y_train)
        
        logger.info("Building XGBoost pipeline with scaling")
        
        try:
            # Create pipeline with standard scaling and XGBoost
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", XGBRegressor(**self.params)),
            ])
            
            logger.info("Training XGBoost model")
            pipeline.fit(X_train, y_train)
            
            logger.info("Model training completed successfully")
            return pipeline
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
    
    @staticmethod
    def _validate_inputs(X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Validate input data."""
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series")
        if X_train.empty or y_train.empty:
            raise ValueError("Training data cannot be empty")
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same length")


class ModelBuilder:
    """
    Context class for model building using the Strategy pattern.
    
    This class allows dynamic switching between different model
    building strategies at runtime.
    """
    
    def __init__(self, strategy: ModelBuildingStrategy):
        """
        Initialize ModelBuilder with a specific strategy.

        Args:
            strategy: The model building strategy to use
        """
        self._strategy = strategy
        logger.info(f"ModelBuilder initialized with {strategy.__class__.__name__}")

    def set_strategy(self, strategy: ModelBuildingStrategy) -> None:
        """
        Set a new model building strategy.

        Args:
            strategy: The new strategy to use
        """
        logger.info(f"Switching to {strategy.__class__.__name__}")
        self._strategy = strategy

    def build_model(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Pipeline:
        """
        Build and train a model using the current strategy.

        Args:
            X_train: Training data features
            y_train: Training data labels/target

        Returns:
            Trained model pipeline

        Raises:
            TypeError: If inputs are invalid
            ValueError: If data is invalid
        """
        logger.info("Building model using current strategy")
        return self._strategy.build_and_train_model(X_train, y_train)


def get_model_strategy(
    model_type: str, 
    params: Optional[Dict[str, Any]] = None
) -> ModelBuildingStrategy:
    """
    Factory function to get the appropriate model strategy.
    
    Args:
        model_type: Type of model ("xgboost" or "linear_regression")
        params: Optional model parameters
    
    Returns:
        Model building strategy instance
    
    Raises:
        ValueError: If model_type is not supported
    """
    strategies = {
        "xgboost": XGBoostStrategy,
        "linear_regression": LinearRegressionStrategy,
    }
    
    if model_type not in strategies:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported types: {list(strategies.keys())}"
        )
    
    return strategies[model_type](params)


if __name__ == "__main__":
    # Example usage
    logger.info("Model building module loaded")

