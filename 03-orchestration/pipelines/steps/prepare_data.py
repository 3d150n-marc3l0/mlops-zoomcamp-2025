from zenml import step
import pandas as pd
from zenml.logger import get_logger
from io import BytesIO

@step(enable_cache=False)
def prepare_features(df, categorical, train=True):
    logger = get_logger(__name__)
    #df['duration'] = df.dropOff_datetime - df.pickup_datetime
    print(f"Records: {df.shape}, Columns: {df.columns}")
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    print(f"Records: {df.shape}, Columns: {df.columns}")
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    print(f"Records: {df.shape}")

    # Save to buffer
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)

    # Size en bytes
    size_bytes = buffer.getbuffer().nbytes
    print(f"Processed Dataset Size in memory: {size_bytes} bytes")

    return df