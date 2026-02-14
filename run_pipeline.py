import click
import mlflow
from pipelines.training_pipeline import ml_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

mlflow.set_tracking_uri(get_tracking_uri())
mlflow.set_experiment("my_experiment_name")


@click.command()
@click.option(
    "--model-type",
    default="xgboost",
    help="Model type to train (xgboost or linear_regression)",
)
def main(model_type: str):
    """
    Run the ML pipeline and start the MLflow UI for experiment tracking.
    """
    # Run the pipeline
    run = ml_pipeline(model_type=model_type)

    # You can uncomment and customize the following lines if you want to retrieve and inspect the trained model:
    # trained_model = run["model_building_step"]  # Replace with actual step name if different
    # print(f"Trained Model Type: {type(trained_model)}")

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the experiment."
    )


if __name__ == "__main__":
    main()
