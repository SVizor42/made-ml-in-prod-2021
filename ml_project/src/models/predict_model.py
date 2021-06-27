import logging
import pickle
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def predict_model(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """
    Makes predictions based on model.
    :param model: the model to predict with
    :param features: the features to predict on
    :return: model predictions
    """
    predictions = model.predict(features)
    return predictions


def deserialize_model(path: str) -> Pipeline:
    """
    Loads model (pipeline) from pickle file.
    :param path: path to file to load from
    :return: deserialized Pipeline
    """
    try:
        with open(path, "rb") as file:
            pipeline = pickle.load(file)
        return pipeline
    except FileNotFoundError as error:
        logger.error(error)
        sys.exit(0)
