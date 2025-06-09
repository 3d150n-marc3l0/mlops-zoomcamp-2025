from zenml import step
from zenml.logger import get_logger
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import numpy as np
import pandas as pd
from typing import Tuple, List

@step(enable_cache=False)
def train_model(df: pd.DataFrame, categorical: List[str]) -> Tuple[LinearRegression, DictVectorizer]:
    logger = get_logger(__name__)
    #categorical = ['PULocationID', 'DOLocationID']
    train_dicts = df[categorical].to_dict(orient='records')
    
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df['duration'].values

    model = LinearRegression()
    model.fit(X_train, y_train)
    logger.info(f"Train Intercept: {model.intercept_}")

    y_pred = model.predict(X_train)
    #rmse = mean_squared_error(y_train, y_pred, squared=False)
    rmse = root_mean_squared_error(y_train, y_pred)
    logger.info(f"Train RMSE: {rmse}")
    logger.info(f"DictVectorizer features: {len(dv.feature_names_)}")

    return model, dv
