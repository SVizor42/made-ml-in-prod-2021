import os
from typing import List

from py._path.local import LocalPath

from src.train_pipeline import train_pipeline
from src.entities import (
    TrainingPipelineParams,
    SplittingParams,
    FeatureParams,
    TrainingParams,
)


def test_train_e2e(
    tmpdir: LocalPath,
    dataset_path: str,
    model_path: str,
    categorical_features: List[str],
    numerical_features: List[str],
    target_col: str,
    features_to_drop: List[str],
):
    expected_metric_path = tmpdir.join("metrics.json")
    params = TrainingPipelineParams(
        input_data_path=dataset_path,
        output_model_path=model_path,
        metric_path=expected_metric_path,
        splitting_params=SplittingParams(val_size=0.2, random_state=42),
        feature_params=FeatureParams(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            target_col=target_col,
            features_to_drop=features_to_drop,
        ),
        train_params=TrainingParams(model_type="RandomForestClassifier"),
    )
    real_model_path, metrics = train_pipeline(params)
    assert metrics["f1_score"] > 0
    assert os.path.exists(real_model_path)
    assert os.path.exists(params.metric_path)
