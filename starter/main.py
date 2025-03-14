"""
FastAPI application for serving ML model predictions.

This module defines a FastAPI application with an endpoint for making predictions using a trained ML model.
"""

from typing import Literal
import pickle
import os
import sys

# Add ML module path to sys.path (to allow import from data)
sys.path.append(os.path.join(os.path.dirname(__file__), "starter", "ml"))

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from data import process_data
import pandas as pd

# Initialize the app
app = FastAPI(
    title="Model API",
    description="This API takes data as input and returns a prediction",
)


class ModelInput(BaseModel):
    """
    Pydantic BaseModel defining the input data schema for the prediction endpoint.

    Attributes:
    - age: Age of the individual.
    - workclass: Working class of the individual.
    - fnlgt: Final weight.
    - education: Education level.
    - education_num: Numeric representation of education level.
    - marital_status: Marital status of the individual.
    - occupation: Occupation of the individual.
    - relationship: Relationship status of the individual.
    - race: Race of the individual.
    - sex: Gender of the individual.
    - capital_gain: Capital gain.
    - capital_loss: Capital loss.
    - hours_per_week: Number of hours worked per week.
    - native_country: Native country of the individual.
    """

    age: int
    fnlgt: int
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    education_num: int
    sex: Literal["Female", "Male"]
    race: Literal[
        "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"
    ]
    education: Literal[
        "Bachelors",
        "HS-grad",
        "11th",
        "Masters",
        "9th",
        "Assoc-voc",
        "Prof-school",
        "5th-6th",
        "10th",
        "Preschool",
        "12th",
        "1st-4th",
        "Some-college",
        "Assoc-acdm",
        "7th-8th",
        "Doctorate",
    ]
    workclass: Literal[
        "Local-gov",
        "Self-emp-inc",
        "Without-pay",
        "State-gov",
        "Self-emp-not-inc",
        "Private",
        "Federal-gov",
    ]
    marital_status: Literal[
        "Never-married",
        "Married-civ-spouse",
        "Married-AF-spouse",
        "Divorced",
        "Married-spouse-absent",
        "Separated",
        "Widowed",
    ]
    occupation: Literal[
        "Tech-support",
        "Craft-repair",
        "Other-service",
        "Transport-moving",
        "Priv-house-serv",
        "Sales",
        "Exec-managerial",
        "Adm-clerical",
        "Farming-fishing",
        "Prof-specialty",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Protective-serv",
        "Armed-Forces",
    ]
    relationship: Literal[
        "Wife",
        "Own-child",
        "Husband",
        "Not-in-family",
        "Other-relative",
        "Unmarried",
    ]

    native_country: Literal[
        "United-States",
        "Cuba",
        "Jamaica",
        "India",
        "Mexico",
        "Puerto-Rico",
        "Honduras",
        "England",
        "Canada",
        "Germany",
        "Iran",
        "Philippines",
        "Poland",
        "Columbia",
        "Cambodia",
        "Thailand",
        "Ecuador",
        "Laos",
        "Taiwan",
        "Haiti",
        "Portugal",
        "Dominican-Republic",
        "El-Salvador",
        "France",
        "Guatemala",
        "Italy",
        "China",
        "South",
        "Japan",
        "Yugoslavia",
        "Peru",
        "Outlying-US(Guam-USVI-etc)",
        "Scotland",
        "Trinadad&Tobago",
        "Greece",
        "Nicaragua",
        "Vietnam",
        "Hong",
        "Ireland",
        "Hungary",
        "Holand-Netherlands",
    ]

    class Config:
        json_schema_extra = {
            "example": {
                "age": 50,
                "capital_gain": 0,
                "capital_loss": 0,
                "education": "Bachelors",
                "education_num": 13,
                "fnlgt": 83311,
                "hours_per_week": 40,
                "marital_status": "Married-civ-spouse",
                "native_country": "United-States",
                "occupation": "Exec-managerial",
                "race": "White",
                "relationship": "Husband",
                "sex": "Male",
                "workclass": "Self-emp-not-inc",
            }
        }


@app.on_event("startup")
async def startup_event():
    """
    Function to be executed on application startup.

    It loads the saved model, encoder, and labelizer objects into global variables.
    """
    loaded_model, loaded_encoder, loaded_lb = load_saved_objects()

    if loaded_model is not None:
        global model, encoder, lb
        model, encoder, lb = loaded_model, loaded_encoder, loaded_lb


def load_saved_objects():
    """
    Function to load the saved model, encoder, and labelizer objects.

    Returns:
    tuple: Trained model, encoder, and labelizer objects.
    """
    model, encoder, lb = None, None, None
    try:
        with open(
            os.path.join("model", "trained_model.pkl"), "rb"
        ) as model_file, open(
            os.path.join("model", "encoder.pkl"), "rb"
        ) as encoder_file, open(
            os.path.join("model", "labelizer.pkl"), "rb"
        ) as lb_file:
            model = pickle.load(model_file)
            encoder = pickle.load(encoder_file)
            lb = pickle.load(lb_file)
    except (FileNotFoundError, EOFError, pickle.PickleError) as e:
        print(f"Error loading saved objects: {e}")

    return model, encoder, lb


@app.get("/")
def read_root():
    """
    Root endpoint to return a welcome message.

    Returns:
    dict: Welcome message.
    """
    return {
        "message": "Hello world! This is the third project of the Udacity MLops Nanodegree!"
    }


@app.post("/predict_salary")
def predict(input: ModelInput):
    """
    Endpoint for making predictions based on input data.

    Args:
    - input (ModelInput): Input data for making predictions.

    Returns:
    dict: Prediction results.
    """
    Input_data = {
        "age": input.age,
        "workclass": input.workclass,
        "fnlgt": input.fnlgt,
        "education": input.education,
        "education-num": input.education_num,
        "marital-status": input.marital_status,
        "occupation": input.occupation,
        "relationship": input.relationship,
        "race": input.race,
        "sex": input.sex,
        "capital-gain": input.capital_gain,
        "capital-loss": input.capital_loss,
        "hours-per-week": input.hours_per_week,
        "native-country": input.native_country,
    }

    # Convert input data into a dataframe
    input_df = pd.DataFrame([Input_data])

    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    # Load saved objects
    with open(
        os.path.join("model", "trained_model.pkl"),
        "rb",
    ) as model_file, open(
        os.path.join("model", "encoder.pkl"), "rb"
    ) as encoder_file, open(
        os.path.join("model", "labelizer.pkl"), "rb"
    ) as lb_file:
        model = pickle.load(model_file)
        encoder = pickle.load(encoder_file)
        lb = pickle.load(lb_file)
    # Process input data
    X, _, _, _ = process_data(
        input_df,
        categorical_features=categorical_features,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    # Make the prediction and process it into a human-readable format
    pred = model.predict(X)
    pred = lb.inverse_transform(pred)[0]
    input_df["prediction"] = pred
    return input_df.to_dict(orient="records")


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
