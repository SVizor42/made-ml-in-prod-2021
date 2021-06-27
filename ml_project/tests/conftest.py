import os
import numpy as np
import pandas as pd
import pytest
from typing import List


TEST_FILE = "train_data_sample.csv"
MODEL_FILE = "model.pkl"
OUTPUT_FILE = "predictions.csv"


@pytest.fixture(scope="module")
def dataset_path():
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, TEST_FILE)
    data = generate_dataset()
    data.to_csv(path)
    return path


@pytest.fixture(scope="module")
def model_path():
    current_dir = os.path.dirname(__file__)
    return os.path.join(current_dir, MODEL_FILE)


@pytest.fixture(scope="module")
def output_data_path():
    current_dir = os.path.dirname(__file__)
    return os.path.join(current_dir, OUTPUT_FILE)


@pytest.fixture(scope="module")
def target_col():
    return "target"


@pytest.fixture(scope="module")
def categorical_features() -> List[str]:
    return [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]


@pytest.fixture(scope="module")
def numerical_features() -> List[str]:
    return [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
    ]


@pytest.fixture(scope="module")
def features_to_drop() -> List[str]:
    return []


def generate_dataset(size: int = 50, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    data = pd.DataFrame()
    data["age"] = np.random.normal(loc=54, scale=9, size=size).astype(int)
    data["sex"] = np.random.binomial(n=1, p=0.7, size=size).astype(int)
    data["cp"] = np.random.randint(low=0, high=4, size=size).astype(int)
    data["trestbps"] = np.random.normal(loc=131, scale=18, size=size).astype(int)
    data["chol"] = np.random.normal(loc=246, scale=52, size=size).astype(int)
    data["fbs"] = np.random.binomial(n=1, p=0.15, size=size).astype(int)
    data["restecg"] = np.random.randint(low=0, high=3, size=size).astype(int)
    data["thalach"] = np.random.normal(loc=150, scale=23, size=size).astype(int)
    data["exang"] = np.random.binomial(n=1, p=0.33, size=size).astype(int)
    data["oldpeak"] = np.clip(np.random.normal(loc=1, scale=2, size=size), 0, None).astype(int)
    data["slope"] = np.random.randint(low=0, high=3, size=size).astype(int)
    data["ca"] = np.random.randint(low=0, high=5, size=size).astype(int)
    data["thal"] = np.random.randint(low=0, high=4, size=size).astype(int)
    data["target"] = np.random.binomial(n=1, p=0.55, size=size).astype(int)
    return data
