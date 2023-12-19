"""
Test the live prediction endpoint on Render
"""
import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")


features = {
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


app_url = "https://udacity-project-3-fdbj.onrender.com/predict_salary"

r = requests.post(app_url, json=features)
assert r.status_code == 200

logging.info("Testing Render app")
logging.info(f"Status code: {r.status_code}")
logging.info(f"Response body: {r.json()}")
