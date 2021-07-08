import json
import logging
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data import read_data, split_train_val_data
from src.entities.train_pipeline_params import (
    TrainingPipelineParams,
    TrainingPipelineParamsSchema,
)
from src.features.build_features import (
    extract_target,
    build_transformer,
    make_features,
)
from src.models import (
    train_model,
    serialize_model,
    predict_model,
    evaluate_model,
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(training_pipeline_params: TrainingPipelineParams):
    """
    The pipeline to transform data, train and evaluate model and store the artifacts.
    :param training_pipeline_params: training parameters
    :return: nothing
    """
    logger.info(f"Start train pipeline with params {training_pipeline_params}.")

    logger.info("Reading data...")
    data = read_data(training_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}.")

    train_df, val_df = split_train_val_data(
        data, training_pipeline_params.splitting_params
    )
    logger.info(f"train_df.shape is {train_df.shape}.")
    logger.info(f"val_df.shape is {val_df.shape}.")

    logger.info("Building transformer...")
    transformer = build_transformer(training_pipeline_params.feature_params)
    logger.info("Fitting transformer...")
    transformer.fit(
        train_df.drop(training_pipeline_params.feature_params.target_col, axis=1)
    )

    logger.info("Preparing train data...")
    train_features = make_features(
        transformer,
        train_df.drop(training_pipeline_params.feature_params.target_col, axis=1)
    )
    train_target = extract_target(train_df, training_pipeline_params.feature_params)
    logger.info(f"train_features.shape is {train_features.shape}.")
    if train_target is not None:
        logger.info(f"train_target.shape is {train_target.shape}.")

    logger.info("Training model...")
    model = train_model(
        train_features, train_target, training_pipeline_params.train_params
    )

    logger.info("Preparing validation data...")
    val_features = make_features(
        transformer,
        val_df.drop(training_pipeline_params.feature_params.target_col, axis=1)
    )
    val_target = extract_target(val_df, training_pipeline_params.feature_params)
    logger.info(f"val_features.shape is {val_features.shape}.")
    if val_target is not None:
        logger.info(f"val_target.shape is {val_target.shape}.")

    logger.info("Making predictions on the validation data...")
    predicts = predict_model(
        model,
        val_features,
    )

    logger.info("Evaluating model...")
    metrics = evaluate_model(
        predicts,
        val_target,
    )
    logger.info(f"Metrics are {metrics}.")

    logger.info("Saving metrics...")
    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)

    logger.info("Saving training pipeline...")
    path_to_model = serialize_model(
        model,
        training_pipeline_params.output_model_path,
        transformer
    )

    logger.info("Done.")
    return path_to_model, metrics


@hydra.main(config_path='../configs', config_name='train_config.yaml')
def train_pipeline_command(config: DictConfig):
    """
    Loads training parameters from config file and starts training process.
    :param config: training configuration
    :return: nothing
    """
    os.chdir(hydra.utils.to_absolute_path('.'))
    schema = TrainingPipelineParamsSchema()
    params = schema.load(config)
    logger.info(f'Training config:\n{OmegaConf.to_yaml(params)}')
    train_pipeline(params)


if __name__ == "__main__":
    train_pipeline_command()
