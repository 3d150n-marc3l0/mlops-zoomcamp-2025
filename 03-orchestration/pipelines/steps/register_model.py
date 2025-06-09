from zenml import step
from zenml.logger import get_logger
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
import os
import pandas as pd
import joblib

@step(experiment_tracker="mlflow_tracker", enable_cache=False)
def register_model(model: LinearRegression, dv: DictVectorizer, df: pd.DataFrame) -> None:
    #with mlflow.start_run():
    logger = get_logger(__name__)

    # Inferir la firma del modelo
    signature = infer_signature(df)

    # Register model
    model_pickle_path = "linear_model.pkl"
    joblib.dump(model, model_pickle_path)
    #mlflow.sklearn.log_model(model, "linear_model")
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="linear_model",
        signature=signature,
        input_example=df.iloc[:1]  # ejemplo de una fila, también es un DataFrame
    )

    # Serialize and log the DictVectorizer
    dv_path = "dict_vectorizer.pkl"
    joblib.dump(dv, dv_path)
    mlflow.log_artifact(dv_path)

    # Log the model_size as an artifact
    model_size_bytes = os.path.getsize(model_pickle_path)
    logger.info(f"Model Size Bytes: {model_size_bytes}")
    mlflow.log_metric('model_size_bytes', model_size_bytes)

    logger.info("Model and DictVectorizer logged to MLflow")
    
    # Remove dv
    os.remove(model_pickle_path)
    os.remove(dv_path)
