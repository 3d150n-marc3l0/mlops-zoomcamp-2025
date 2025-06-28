
import pandas as pd
from datetime import datetime
from batch import prepare_data 



def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_prepare_data():
    # Create data test
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]
    
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    # Preprocess
    pre_df = prepare_data(df)
    #df['ride_id'] = df.index.astype('str')
    print(pre_df)

    # Expected data
    expected_columns = ['PULocationID', 'DOLocationID', 'duration']
    expected_data = [
        ('-1', '-1', 9.0),
        ('1', '1', 8.0),
    ]
    expected_df = pd.DataFrame(expected_data, columns=expected_columns)
    print(expected_df)

    # Test
    pd.testing.assert_frame_equal(pre_df[expected_columns].reset_index(drop=True), expected_df)
