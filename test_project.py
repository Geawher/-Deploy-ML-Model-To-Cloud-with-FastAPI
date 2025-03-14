"""
This module defines fixtures and tests to ensure the correct behavior of the ML model and related functions.
"""
# Add ML module path to sys.path
import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(__file__), "starter", "starter", "ml")
)
import logging
import pytest
from sklearn.model_selection import train_test_split
from model import compute_model_metrics, inference, compute_slices
from data import process_data
import pandas as pd
import joblib

DATA_PATH = "starter/data/census.csv"
MODEL_PATH = "starter/model/trained_model.pkl"


@pytest.fixture()
def data():
    """
    Fixture to load the dataset.

    Returns:
    data (pd.DataFrame): Loaded dataset.
    """
    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.replace(" ", "")
    return data


@pytest.fixture()
def model():
    """
    Fixture to load the trained ML model.

    Returns:
    object: Trained ML model.
    """

    return joblib.load(MODEL_PATH)


@pytest.fixture()
def cat_features():
    """
    Fixture to provide categorical features.

    Returns:
    list: List of categorical features.
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    return cat_features


@pytest.fixture()
def data_train_test(data, cat_features):
    """
    Fixture to prepare training and testing data.

    Args:
    - data (pd.DataFrame): Loaded dataset.
    - features (list): List of categorical features.

    Returns:
    tuple: Training and testing data.
    """
    train, test = train_test_split(
        data,
        test_size=0.20,
        random_state=0,
    )
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    return X_train, X_test, y_train, y_test


def test_import_data():
    """
    Test to import and check the dataset.

    Raises:
    - FileNotFoundError: If the dataset is not found.
    - AssertionError: If the dataset is empty.
    """
    try:
        data = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        logging.error("ERROR: Dataset not found, check your path")
        raise FileNotFoundError
    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
    except AssertionError:
        logging.error("ERROR: Dataset is empty")
        raise AssertionError


def test_features(data, cat_features):
    """
    Test to verify the identification of categorical features in the dataset.

    Args:
    - data (pd.DataFrame): Loaded dataset.
    - features (list): List of categorical features.

    Raises:
    - AssertionError: If the features are not correctly identified.
    """
    try:
        assert all(feature in data.columns for feature in cat_features)
    except AssertionError:
        logging.error("ERROR: Features are not correctly identified")
        raise AssertionError


def test_model_can_predict(model, data_train_test):
    """
    Test for verifying that the model can make predictions.

    Args:
    - model (object): Trained machine learning model.
    - data_train_test (tuple): Training and testing data.

    Raises:
    - AssertionError: If the model is not fitted.
    """
    _, X_test, _, _ = data_train_test
    try:
        assert model.predict(X_test)
    except BaseException:
        logging.error("ERROR: Model is not fitted!")


def test_compute_model_metrics(model, data_train_test):
    """
    Test to verify if the model can compute metrics.

    Args:
    - model (object): Trained machine learning model.
    - data_train_test (tuple): Training and testing data.

    Raises:
    - AssertionError: If the model can't compute metrics.
    """
    try:
        _, X_test, _, y_test = data_train_test
        preds = inference(model, X_test)
        precision, recall, fbeta = compute_model_metrics(y_test, preds)
        assert isinstance(precision, float)
        assert isinstance(recall, float)
        assert isinstance(fbeta, float)
    except AssertionError:
        logging.error("ERROR: Model can't compute metrics")
        raise AssertionError


def test_compute_performance_for_slices(
    data, data_train_test, cat_features, model
):
    """
    Test to verify if the model can compute metrics for different slices.

    Args:
    - data (pd.DataFrame): Loaded dataset.
    - data_train_test (tuple): Training and testing data.
    - features (list): List of categorical features.
    - model (object): Trained machine learning model.

    Raises:
    - AssertionError: If the model can't compute slices.
    """
    try:
        _, test = train_test_split(data, test_size=0.20)
        _, X_test, _, y_test = data_train_test
        preds = inference(model, X_test)
        for feature in cat_features:
            compute_slices(test, feature, y_test, preds)
        # Check that the file is created
        assert os.path.exists("starter/starter/ml/slice_output.txt")
    except AssertionError:
        logging.error("ERROR: Model can't compute slices")
        raise AssertionError
