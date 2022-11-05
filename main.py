# Put the code for your API here.
import os

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from fastapi.encoders import jsonable_encoder
import json

import pandas as pd
import logging

from starter.code.predict import predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

class CensusData(BaseModel):
    age: float
    workclass: str
    fnlgt: float
    education: str
    education_num: float
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float
    capital_loss: float
    hours_per_week: float
    native_country: str

    class Config:
        schema_extra = {
            "example": {
                "age": 52,
                "workclass": "Self-emp-not-inc",
                "fnlgt": 209642,
                "education": "HS-grad",
                "education_num": 9,
                "marital_status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 45,
                "native_country": "United-States",
            }
        }


app = FastAPI()

@app.get("/")
async def greeting():
    return {"message": "Welcome to Udacity's 3rd Project!"}

@app.post("/predict")
async def make_predictions(record: CensusData):
    logger.info("The post request data: %s", record)
    input_data = jsonable_encoder(record)
    input_data = pd.DataFrame(input_data, index=[0])
    logger.info(f"Input data: {input_data}")
    result = predict(input_data)
    logger.info("The predicted result is %s", result)
    return result
