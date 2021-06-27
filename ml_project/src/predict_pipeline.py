import logging
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data import read_data
from src.entities.predict_pipeline_params import (
    PredictPipelineParams,
    PredictPipelineParamsSchema,
)
from src.models import (
    deserialize_model,
    predict_model,
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def predict_pipeline(predict_pipeline_params: PredictPipelineParams):
    """
    The pipeline to load model from file and start prediction on given data.
    :param predict_pipeline_params: prediction parameters
    :return: nothing
    """
    logger.info(f"Start prediction pipeline with params {predict_pipeline_params}.")

    logger.info("Reading data...")
    data = read_data(predict_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}.")

    logger.info("Loading pipeline...")
    pipeline = deserialize_model(predict_pipeline_params.model_path)

    logger.info("Making predictions on the provided data...")
    predicts = predict_model(pipeline, data)

    logger.info("Saving predictions...")
    data["predictions"] = predicts
    data.to_csv(predict_pipeline_params.output_data_path)
    logger.info("Done.")


@hydra.main(config_path='../configs', config_name='predict_config.yaml')
def predict_pipeline_command(config: DictConfig):
    """
    Loads prediction parameters from config file and starts the prediction process.
    :param config: prediction configuration
    :return: nothing
    """
    os.chdir(hydra.utils.to_absolute_path('.'))
    schema = PredictPipelineParamsSchema()
    params = schema.load(config)
    logger.info(f'Prediction config:\n{OmegaConf.to_yaml(params)}')
    predict_pipeline(params)


if __name__ == "__main__":
    predict_pipeline_command()
