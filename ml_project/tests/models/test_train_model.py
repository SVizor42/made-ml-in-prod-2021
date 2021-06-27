import os
import pickle
from typing import List, Tuple

import pandas as pd
import pytest
from py._path.local import LocalPath
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.data.make_dataset import read_data
from src.entities import TrainingParams
from src.entities.feature_params import FeatureParams
from src.features.build_features import make_features, extract_target, build_transformer
from src.models.train_model import train_model, serialize_model


@pytest.fixture(scope="function")
def features_and_target(
    dataset_path: str, categorical_features: List[str], numerical_features: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        features_to_drop=[],
        target_col="target",
    )
    data = read_data(dataset_path)
    transformer = build_transformer(params)
    transformer.fit(data)
    features = make_features(transformer, data)
    target = extract_target(data, params)
    return features, target


def test_train_model(features_and_target: Tuple[pd.DataFrame, pd.Series]):
    features, target = features_and_target
    model = train_model(features, target, train_params=TrainingParams())
    assert isinstance(model, RandomForestClassifier)
    assert model.predict(features).shape[0] == target.shape[0]


def test_serialize_model(tmpdir: LocalPath):
    expected_output = tmpdir.join("model.pkl")
    transformer = ColumnTransformer([])
    n_estimators = 10
    max_depth = 4
    model = RandomForestClassifier(n_estimators=gitn_estimators, max_depth=max_depth)
    real_output = serialize_model(model, expected_output, transformer)
    assert real_output == expected_output
    assert os.path.exists
    with open(real_output, "rb") as file:
        model = pickle.load(file)
    assert isinstance(model, Pipeline)
