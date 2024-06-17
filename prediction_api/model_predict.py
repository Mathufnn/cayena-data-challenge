from catboost import CatBoostClassifier, Pool
from pydantic_models import PredictionInput
import pandas as pd

def load_model(model_path: str):
    """Load the CatBoost model from the specified file."""
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model

def transform_data_features(data: PredictionInput):
    confirm_minute = pd.to_datetime(data["date_time_confirm"]).minute
    confirm_second = pd.to_datetime(data["date_time_confirm"]).second

    time_delta = pd.to_datetime(data["date_time_confirm"]) - pd.to_datetime(data["date_time_login"])
    delta_seconds = time_delta.total_seconds()

    col_5_7 = data['col_5'] * data['col_7']

    features = [
        data['col_0'], data['col_1'], data['col_2'], data['col_3'],
        data['col_4'], data['col_5'], data['col_6'], data['col_7'],
        data['FU'], data['City'], data['CEP'], confirm_minute,
        confirm_second, delta_seconds, col_5_7
    ]

    return features


def predict(model: CatBoostClassifier, data: dict):
    """Make a prediction using the provided model and input data."""
    features = transform_data_features(data=data)

    input_pool = Pool(
        data=[features],
        cat_features=[8, 9]  # indices of the categorical features ["FU", "City"]
    )

    prediction = model.predict(input_pool)
    probability = model.predict_proba(input_pool).max()

    return int(prediction[0]), float(probability)


def predict_batch(model: CatBoostClassifier, data_list: list):
    """Make batch predictions using the provided model and input data."""
    features_list = [transform_data_features(data=data) for data in data_list]

    input_pool = Pool(
        data=features_list,
        cat_features=[8, 9]  # indices of the categorical features ["FU", "City"]
    )

    predictions = model.predict(input_pool)
    probabilities = model.predict_proba(input_pool)

    results = [(int(pred), float(prob.max())) for pred, prob in zip(predictions, probabilities)]

    return results