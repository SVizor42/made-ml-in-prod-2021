import yaml
from dataclasses import dataclass
from marshmallow_dataclass import class_schema

from .split_params import SplittingParams
from .feature_params import FeatureParams
from .train_params import TrainingParams


@dataclass()
class TrainingPipelineParams:
    """
    Dataclass for training pipeline configuration.
    """
    input_data_path: str
    output_model_path: str
    metric_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    """
    Reads configuration file for training pipeline.
    :param path: path to configuration file
    :return: configuration dataclass
    """
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
