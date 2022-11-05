import sys
sys.path.append('../udacity-project3')

from fastapi.testclient import TestClient
from fastapi.encoders import jsonable_encoder

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to Udacity's 3rd Project!"}

def test_predictions_pos():
    data = {
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
    "native_country": "United-States",
    }

    r = client.post("/predict",json=data)

    assert r.status_code == 200
    assert r.json() == '{"prediction": [" <=50K"]}'

def test_predictions_neg():
    data = {"age": 31,
    "workclass": "Private",
    "fnlgt": 45781,
    "education": "Masters",
    "education_num": 14,
    "marital_status": "Never-married",
    "occupation": "Prof-specialty",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Female",
    "capital_gain": 14084,
    "capital_loss": 0,
    "hours_per_week": 50,
    "native_country": "United-States"}

    r = client.post("/predict",json=data)

    assert r.status_code == 200
    assert r.json() == '{"prediction": [" >50K"]}'
