import logging
from typing import Annotated

import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from src.model_building import LinearRegressionStrategy, XGBoostStrategy
from zenml import ArtifactConfig, step, Model
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker

# Initialize the experiment tracker
# experiment_tracker = MLFlowExperimentTracker(
#     name="mlflow_tracker",
#     tracking_uri="file:./mlruns"
# )


model = Model(
    name="prices_predictor",
    version=None,
    license="Apache 2.0",
    description="Price prediction model for houses.",
)


@step(enable_cache=False, experiment_tracker="mlflow_tracker")
def model_building_step(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "xgboost",
) -> Annotated[Pipeline, ArtifactConfig(name="sklearn_pipeline", is_model_artifact=True)]:
    """
    Builds and trains a model using the selected strategy.

    Parameters:
    X_train (pd.DataFrame): The training data features.
    y_train (pd.Series): The training data labels/target.
    model_type (str): The type of model to train ("linear_regression" or "xgboost").

    Returns:
    Pipeline: The trained pipeline including preprocessing and the model.
    """
    # Ensure the inputs are of the correct type
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

    logging.info(f"Categorical columns: {categorical_cols.tolist()}")
    logging.info(f"Numerical columns: {numerical_cols.tolist()}")

    # Define preprocessing for categorical and numerical features
    numerical_transformer = SimpleImputer(strategy="mean")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        verbose_feature_names_out=False,
    )
    
    # Configure preprocessor to return pandas DataFrame
    preprocessor.set_output(transform="pandas")

    # Start an MLflow run to log the model training process
    if not mlflow.active_run():
        mlflow.start_run()  # Start a new MLflow run if there isn't one active

    try:
        # Enable autologging
        mlflow.sklearn.autolog()
        mlflow.xgboost.autolog()

        logging.info("Preprocessing data.")
        X_train_transformed = preprocessor.fit_transform(X_train, y_train)

        logging.info(f"Selecting model strategy: {model_type}")
        if model_type == "linear_regression":
            strategy = LinearRegressionStrategy()
        elif model_type == "xgboost":
            strategy = XGBoostStrategy()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        logging.info("Building and training the model.")
        # Train the model using the strategy (which includes scaling)
        trained_model_pipeline = strategy.build_and_train_model(X_train_transformed, y_train)
        
        logging.info("Model training completed.")

        # Create the full pipeline
        full_pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", trained_model_pipeline)]
        )

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e

    finally:
        # End the MLflow run
        mlflow.end_run()

    return full_pipeline
