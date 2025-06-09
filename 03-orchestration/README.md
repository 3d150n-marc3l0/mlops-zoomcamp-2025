# Project

## Overview

Within the project we have the following file structure:

```plaintext
```plaintext
.
├── pipelines                   
│   ├── run_pipeline.py        # Pipeline.
│   └── steps
│       ├── load_data.py       # Download dataset.
│       ├── prepare_data.py    # Preprocessing.
│       ├── register_model.py  # Register model to mlflow.
│       └── train_model.py     # Train model.
└── zenml                       
    ├── docker-compose.yml     # Docker-compose with services: zenml (orquestator) and mflow (tracker).
    └── mlflow.dockerfile      # Mlfow Docker file.
```

* [run_pipeline.py](pipelines/run_pipeline.py). This module contains the logic to execute the end-to-end pipeline.
* [load_data.py](pipelines/steps/load_data.py). This module contains the logic to download the NYC taxi dataset and returns a pandas dataframe.
* [prepare_data.py](pipelines/steps/prepare_data.py). This module contains the logic to perform preprocessing on the pandas dataframe.
* [train_model.py](pipelines/steps/train_model.py). This module contains the logic for training a linear regression model.
* [register_model.py](pipelines/steps/register_model.py). This module contains the logic to register the trained model to the mlflow tracker.
* [docker-compose.yml](zenml/docker-compose.yml). This file contains the specification of the services to be able to run the zenml orchestrator and mlflow tracker.
* [mlflow.dockerfile](zenml/mlflow.dockerfile). Docker file containing the container definition to run mlflow.


## Init project

```bash
zenml init
```

The zenml init command creates a .zen directory, which will store your pipeline configurations and local database.

## Login
```bash
zenml login http://localhost:8080
```

## Mlflow Setting

We need to install it on your local machine to be able to register an MLflow Experiment Tracker and add it to your stack:

```bash
zenml integration install mlflow -y
```

Register the MLflow experiment tracker

```bash
zenml experiment-tracker register mlflow_tracker \
  --flavor=mlflow \
  --tracking_uri=http://localhost:5000 \
  --tracking_username=dummy \
  --tracking_password=dummy
```

Register and set a stack with the new experiment tracker

```bash
zenml stack register mlflow_stack  \
    -a default \
    -o default \
    -e mlflow_tracker \
    --set
```

## Execute Pipeline

```bash
cd pipelines
python run_pipeline.py --dataset_type yellow --year 2023 --month 3
```