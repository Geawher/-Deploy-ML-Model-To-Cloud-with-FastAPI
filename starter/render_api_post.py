"""
Test the live prediction endpoint on Render
"""
import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")


features = {
    "age": 36,
    "workclass": "Private",
    "fnlgt": 302146,
    "education": "HS-grad",
    "education_num": 9,
    "marital_status": "Divorced",
    "occupation": "Craft-repair",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 2000,
    "capital_loss": 0,
    "hours_per_week": 45,
    "native_country": "United-States",
}


app_url = "https://udacity-project-3-fdbj.onrender.com/predict_salary"

r = requests.post(app_url, json=features)
assert r.status_code == 200

logging.info("Testing Render app")
logging.info(f"Status code: {r.status_code}")
logging.info(f"Response body: {r.json()}")
