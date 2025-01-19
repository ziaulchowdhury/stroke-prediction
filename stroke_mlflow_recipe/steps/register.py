from mlflow import log_artifact, register_model

def register():
    #model_path = config("train")["model"]
    #register_model(model_uri=model_path, name="stroke_model")
    print("Registering MLFlow Recipe")
