import mlflow
import joblib
from mlflow.recipes.utils.execution import get_step_output

def register():
    # Load the trained model from the train step
    model_path = get_step_output("train")["model"]
    model = joblib.load(model_path)

    # Define the model name for the registry
    model_name = "regression_model_example"

    # Start an MLflow run to register the model
    with mlflow.start_run():
        # Log the model to MLflow
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=model_name)

    print(f"Model registered to the MLflow Model Registry under the name: {model_name}")
