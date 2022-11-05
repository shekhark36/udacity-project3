import joblib
import os

import pandas as pd
import numpy as np
import json

from starter.code.data import clean_data, process_data
from starter.code.model import inference

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]


def predict(data):
    scaler = joblib.load(os.path.join("starter", "code", "scaler.pkl"))
    encoder = joblib.load(os.path.join("starter", "code","encoder.pkl"))
    lg_model = joblib.load(os.path.join("starter", "code","logistic_model.pkl"))
    lb = joblib.load(os.path.join("starter", "code","label_encoder.pkl"))

    data = clean_data(data)

    X_test, _, encoder, scaler, lb = process_data(
        data, categorical_features=cat_features, label=None, training=False,
        encoder=encoder, scaler=scaler, lb=lb)
    
    preds = lb.inverse_transform(inference(lg_model, X_test))
    return json.dumps({"prediction": preds.tolist()})

if __name__ == "__main__":
    data = """[{
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
    }]"""

    data = pd.read_json(data, orient='list')
    print(predict(data))
