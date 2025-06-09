from zenml import step
from zenml.logger import get_logger
import pandas as pd
import requests
from io import BytesIO

@step
def load_data(dataset_type: str, year: int, month: int) -> pd.DataFrame:
    logger = get_logger(__name__)
    if dataset_type not in {"yellow", "green"}:
        raise ValueError("dataset_type must be 'yellow' or 'green'")

    month_str = f"{month:02d}"
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{dataset_type}_tripdata_{year}-{month_str}.parquet"
    
    logger.info(f"url: {url}")

    #url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
    #df = pd.read_parquet(url)

    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(response.text)

    df = pd.read_parquet(BytesIO(response.content))

    print(f"Records in {dataset_type}_tripdata_{year}-{month_str}: {df.shape}")
    return df

