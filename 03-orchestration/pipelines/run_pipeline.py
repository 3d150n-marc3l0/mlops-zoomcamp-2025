from zenml import pipeline
from steps.load_data import load_data
from steps.prepare_data import  prepare_features
from steps.train_model import train_model
from steps.register_model import register_model
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run Taxi ML pipeline with ZenML")
    parser.add_argument("--dataset_type", type=str, choices=["yellow", "green"], required=True, help="Type of taxi dataset to use")
    parser.add_argument("--year", type=int, required=True, help="Year of the dataset")
    parser.add_argument("--month", type=int, required=True, help="Month of the dataset (1-12)")
    return parser.parse_args()


@pipeline
def taxi_training_pipeline(dataset_type: str, year: int, month: int):
    # Load data
    df = load_data(
        dataset_type=dataset_type,
        year=year,
        month=month
    )

    # Prepare data
    categorical = ['PULocationID', 'DOLocationID']
    df_train_processed = prepare_features(df, categorical)

    # Train model
    model, dv = train_model(df_train_processed, categorical)

    # Register model
    register_model(model, dv, df_train_processed)

if __name__ == "__main__":
    # python run_pipeline.py --dataset_type yellow --year 2023 --month 3

    args = parse_args()
    categorical = ["PULocationID", "DOLocationID"]
    
    taxi_training_pipeline(
        dataset_type=args.dataset_type,
        year=args.year,
        month=args.month
    )