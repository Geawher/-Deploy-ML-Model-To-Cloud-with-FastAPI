"""
Tests for the FastAPI application.

Uses pytest and FastAPI TestClient for testing the API endpoints.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))

from main import app
import logging
import pytest
from fastapi.testclient import TestClient

client = TestClient(app)


@pytest.fixture
def sample_data():
    """
    Fixture providing a sample data dictionary for testing for class 1 (<=50K).

    Returns:
    dict: A dictionary containing sample input data.
    """
    return {
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


@pytest.fixture
def sample_data2():
    """
    Fixture providing a sample data dictionary for testing for class 0 (>50K).

    Returns:
    dict: A dictionary containing sample input data.
    """
    return {
        "age": 52,
        "capital_gain": 15024,
        "capital_loss": 0,
        "education": "HS-grad",
        "education_num": 9,
        "fnlgt": 287927,
        "hours_per_week": 40,
        "marital_status": "Married-civ-spouse",
        "native_country": "United-States",
        "occupation": "Exec-managerial",
        "race": "White",
        "relationship": "Wife",
        "sex": "Female",
        "workclass": "Self-emp-inc",
    }


def test_welcome_message():
    """
    Test for the welcome message endpoint ("/").

    Checks if the response status code is 200 and if the returned message matches the expected message.
    """
    r = client.get("/")
    assert r.status_code == 200
    assert (
        r.json()["message"]
        == "Hello world! This is the third project of the Udacity MLops Nanodegree!"
    )


def test_model_inference_class1(sample_data):
    """
    Test for model inference endpoint ("/predict") for class 1 prediction.

    Checks if the response status code is 200 and if the returned prediction matches the expected values.
    """
    r = client.post("/predict", json=sample_data)

    assert r.status_code == 200
    assert r.json()[0]["age"] == sample_data["age"]
    assert r.json()[0]["fnlgt"] == sample_data["fnlgt"]
    assert r.json()[0]["prediction"] == " <=50K"


def test_model_inference_class_0(sample_data2):
    """
    Test for model inference endpoint ("/predict") for class 0 prediction.
    """

    r = client.post("/predict", json=sample_data2)
    assert r.status_code == 200
    assert r.json()[0]["age"] == sample_data2["age"]
    assert r.json()[0]["fnlgt"] == sample_data2["fnlgt"]
    assert r.json()[0]["prediction"] == " >50K"


def test_incomplete_inference_query():
    """
    Test for incomplete model inference query.

    Checks if the response status code is 422 and if the 'prediction' key is not present in the response.
    """
    data = {
        "occupation": "Prof-specialty",
        "race": "Black",
        "fnlgt": 5178,
        "sex": "Female",
    }
    r = client.post("/predict", json=data)
    assert r.status_code == 422
    assert "prediction" not in r.json()["detail"][0].keys()

    logging.warning(
        f"The sample has {len(data)} features. Must be 14 features"
    )
