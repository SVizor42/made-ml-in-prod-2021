import pytest
import random
import sys
import logging
from typing import List
from fastapi.testclient import TestClient

from app import app, load_model
from src.entities import (
    MAIN_ENDPOINT_MSG,
    FEATURE_NAMES,
    FEATURES_MIN_MAX,
    N_FEATURES,
    INCORRECT_FEATURES_MSG
)

client = TestClient(app)
load_model()

HEALTH_ENDPOINT_MSG = "Pipeline is ready: True"
N_SAMPLES = 10

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def send_request(data: List, features: List):
    request = {
        "data": data,
        "features": features,
    }
    return request


@pytest.fixture(scope="module")
def empty_data():
    samples = [[]]
    return send_request(samples, FEATURE_NAMES)


@pytest.fixture(scope="module")
def data():
    samples = [
        [
            random.randint(min_val, max_val)
            if (min_val is not None) and (max_val is not None)
            else random.random()
            for (_, min_val, max_val) in FEATURES_MIN_MAX
        ]
        for _ in range(N_SAMPLES)
    ]
    return send_request(samples, FEATURE_NAMES)


@pytest.fixture(scope="module")
def data_with_bad_types():
    samples = [["str"] * N_FEATURES] * N_SAMPLES
    return send_request(samples, FEATURE_NAMES)


@pytest.fixture(scope="module")
def data_with_cat_vals_oor():
    samples = [
        [
            random.randint(max_val, max_val + 1)
            if (min_val is not None) and (max_val is not None)
            else random.random()
            for (_, min_val, max_val) in FEATURES_MIN_MAX
        ]
        for _ in range(N_SAMPLES)
    ]
    return send_request(samples, FEATURE_NAMES)


@pytest.fixture(scope="module")
def data_with_bad_column_order(data):
    samples = [sample for sample in data["data"]]
    return send_request(samples, FEATURE_NAMES[::-1])


@pytest.fixture(scope="module")
def data_without_column(data):
    samples = [sample[1:] for sample in data["data"]]
    return send_request(samples, FEATURE_NAMES)


@pytest.fixture(scope="module")
def data_with_extra_columns(data):
    samples = [
        sample[:1] + sample for sample in data["data"]
    ]
    return send_request(samples, FEATURE_NAMES)


@pytest.fixture(scope="module")
def features_without_column(data):
    samples = data["data"]
    return send_request(samples, FEATURE_NAMES[1:])


@pytest.fixture(scope="module")
def features_with_extra_columns(data):
    samples = data["data"]
    return send_request(
        samples, FEATURE_NAMES[:1] + FEATURE_NAMES
    )


@pytest.fixture(scope="module")
def data_features_without_column(data):
    samples = [sample[1:] for sample in data["data"]]
    return send_request(samples, FEATURE_NAMES[1:])


@pytest.fixture(scope="module")
def data_features_with_extra_columns(data):
    samples = [
        sample[:1] + sample for sample in data["data"]
    ]
    return send_request(
        samples, FEATURE_NAMES[:1] + FEATURE_NAMES
    )


@pytest.fixture(scope="module")
def data_without_column_features_with_extra_one(data):
    samples = [sample[1:] for sample in data["data"]]
    return send_request(
        samples, FEATURE_NAMES[:1] + FEATURE_NAMES
    )


@pytest.fixture(scope="module")
def data_with_extra_columns_features_without(data):
    samples = [
        sample[:1] + sample for sample in data["data"]
    ]
    return send_request(samples, FEATURE_NAMES[1:])


def test_main():
    with TestClient(app) as c:
        response = c.get("/")
        assert 200 == response.status_code
        assert MAIN_ENDPOINT_MSG == response.json()


def test_health():
    response = client.get("/health")
    assert 200 == response.status_code
    assert HEALTH_ENDPOINT_MSG == response.json()


def test_predict_without_data(empty_data):
    response = client.get("/predict/", json=empty_data)
    assert 422 == response.status_code
    assert f"ensure this value has at least {N_FEATURES} items" \
           == response.json()["detail"][0]["msg"]


def test_predict(data):
    response = client.get("/predict/", json=data)
    assert 200 == response.status_code
    assert N_SAMPLES == len(response.json())


def test_predict_bad_types(data_with_bad_types):
    response = client.get("/predict/", json=data_with_bad_types)
    assert 422 == response.status_code
    assert all(
        "value is not a valid integer" == item["msg"]
        for item in response.json()["detail"][::2]
    )
    assert all(
        "value is not a valid float" == item["msg"]
        for item in response.json()["detail"][1::2]
    )


def test_predict_with_cat_values_out_of_range(data_with_cat_vals_oor):
    response = client.get("/predict/", json=data_with_cat_vals_oor)
    assert 422 == response.status_code
    assert 1 == len(response.json()["detail"])
    assert all(
        "Feature value is out of range." == item["msg"]
        for item in response.json()["detail"]
    )


def test_predict_incorrect_column_order(data_with_bad_column_order):
    response = client.get("/predict/", json=data_with_bad_column_order)
    assert 422 == response.status_code
    assert 1 == len(response.json()["detail"])
    assert all(
        "Incorrect features order or columns" == item["msg"]
        for item in response.json()["detail"]
    )


def test_predict_incorrect_data_without_column(data_without_column):
    response = client.get("/predict/", json=data_without_column)
    assert 422 == response.status_code
    assert N_SAMPLES == len(response.json()["detail"])
    assert all(
        f"ensure this value has at least {N_FEATURES} items"
        == item["msg"] for item in response.json()["detail"]
    )


def test_predict_incorrect_data_with_extra_columns(data_with_extra_columns):
    response = client.get("/predict/", json=data_with_extra_columns)
    assert 422 == response.status_code
    assert N_SAMPLES == len(response.json()["detail"])
    assert all(
        f"ensure this value has at most {N_FEATURES} items"
        == item["msg"] for item in response.json()["detail"]
    )


def test_predict_incorrect_features_without_column(features_without_column):
    response = client.get("/predict/", json=features_without_column)
    assert 422 == response.status_code
    assert 1 == len(response.json()["detail"])
    assert all(
        INCORRECT_FEATURES_MSG == item["msg"]
        for item in response.json()["detail"]
    )


def test_predict_incorrect_features_with_extra_columns(
        features_with_extra_columns
):
    response = client.get("/predict/", json=features_with_extra_columns)
    assert 422 == response.status_code
    assert 1 == len(response.json()["detail"])
    assert all(
        INCORRECT_FEATURES_MSG == item["msg"]
        for item in response.json()["detail"]
    )


def test_predict_incorrect_data_features_without_column(
        data_features_without_column
):
    response = client.get("/predict/", json=data_features_without_column)
    assert 422 == response.status_code
    assert N_SAMPLES + 1 == len(response.json()["detail"])
    assert all(
        f"ensure this value has at least {N_FEATURES} items" == item["msg"]
        for item in response.json()["detail"][:N_SAMPLES]
    )
    assert INCORRECT_FEATURES_MSG \
           == response.json()["detail"][N_SAMPLES]["msg"]


def test_predict_incorrect_data_features_with_extra_columns(
        data_features_with_extra_columns
):
    response = client.get("/predict/", json=data_features_with_extra_columns)
    assert 422 == response.status_code
    assert N_SAMPLES + 1 == len(response.json()["detail"])
    assert all(
        f"ensure this value has at most {N_FEATURES} items" == item["msg"]
        for item in response.json()["detail"][:N_SAMPLES]
    )
    assert INCORRECT_FEATURES_MSG \
           == response.json()["detail"][N_SAMPLES]["msg"]


def test_predict_incorrect_data_without_column_features_with(
        data_without_column_features_with_extra_one
):
    response = client.get(
        "/predict/", json=data_without_column_features_with_extra_one
    )
    assert 422 == response.status_code
    assert N_SAMPLES + 1 == len(response.json()["detail"])
    assert all(
        f"ensure this value has at least {N_FEATURES} items" == item["msg"]
        for item in response.json()["detail"][:N_SAMPLES]
    )
    assert INCORRECT_FEATURES_MSG \
           == response.json()["detail"][N_SAMPLES]["msg"]


def test_predict_incorrect_data_with_extra_columns_features_without(
        data_with_extra_columns_features_without
):
    response = client.get(
        "/predict/", json=data_with_extra_columns_features_without
    )
    assert 422 == response.status_code
    assert N_SAMPLES + 1 == len(response.json()["detail"])
    assert all(
        f"ensure this value has at most {N_FEATURES} items" == item["msg"]
        for item in response.json()["detail"][:N_SAMPLES]
    )
    assert INCORRECT_FEATURES_MSG \
           == response.json()["detail"][N_SAMPLES]["msg"]
