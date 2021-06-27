import os
from typing import List

from py._path.local import LocalPath

from src.predict_pipeline import predict_pipeline
from src.entities import (
    PredictPipelineParams,
    FeatureParams,
)


def test_train_e2e(
    tmpdir: LocalPath,
    dataset_path: str,
    output_data_path: str,
    model_path: str,
    categorical_features: List[str],
    numerical_features: List[str],
    target_col: str,
    features_to_drop: List[str],
):
    expected_metric_path = tmpdir.join("metrics.json")
    params = PredictPipelineParams(
        input_data_path=dataset_path,
        output_data_path=output_data_path,
        model_path=model_path,
        metric_path=expected_metric_path,
        feature_params=FeatureParams(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            target_col=target_col,
            features_to_drop=features_to_drop,
        ),
    )
    predict_pipeline(params)
    assert os.path.exists(params.output_data_path)
