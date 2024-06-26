# Cayena Data Science Challenge

## Overview

This document outlines the file structure and functionality of the Cayena Data Science Challenge, which is divided into two parts: Predictive Analysis and Model Deployment.

## 1. Predictive Analysis

- **data_analysis.ipynb**: The primary Jupyter notebook containing the main predictive analysis, data exploration, and model training.
- **locations_retriever.ipynb**: Jupyter notebook with a script that sends requests to the "cep.awesomeapi" to retrieve location data based on CEP, used as a feature for models.
- **dataset_ds_case.csv**: Dataset used for the main analysis (NOT synced to github to maintain dataset privacy)
- **locations_data.csv**: Location data generated by the `locations_retriever.ipynb` file.

## 2. Model Deployment

The files required for model deployment are contained within the "prediction_api" folder. This folder allows you to run a Docker container that hosts an API serving the model.

### Running the Docker Container (Linux)

To run the Docker container, execute the following commands:

```sh
cd path/to/folder/prediction_api
docker build -t cayena-prediction-api .
docker run -d -p 8000:8000 cayena-prediction-api
```

These commands will start a container running the prediction API server. 

### Getting Model Predictions

To get model prediction + probability, call the API as shown in the following example:

```sh
curl --location --request POST 'http://localhost:8000/predict' \
--header 'Content-Type: application/json' \
--data-raw '{
    "col_0": -880.747368,
    "col_1": -956.608685,
    "col_2": 1254.880642,
    "col_3": -295.771103,
    "col_4": -4279.430521,
    "col_5": -1540.310595,
    "col_6": -376.902843,
    "col_7": -550.748455,
    "FU": "RJ",
    "City": "Rio de Janeiro",
    "CEP": 23580304,
    "confirm_minute": 4,
    "confirm_second": 52,
    "delta_seconds": 16.0,
    "col_5_7": 848323.680021
}'
```

Finally, to stop the docker container, simply run the following:

```sh
docker ps
docker stop container_id
```