import logging
import os
import sys
import pandas as pd
import numpy as np
import pickle
import click
import json
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from typing import Dict

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

VAL_FILENAME = "val.csv"
MODEL_FILENAME = "model.pkl"
TARGET_COL = "target"
METRIC_FILENAME = "metrics.json"


def load_model(path: str):
    """
    Loads model from pickle file.
    :param path: path to file to load from
    :return: deserialized model
    """
    try:
        with open(path, "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError as error:
        logger.error(error)
        sys.exit(0)


def evaluate_model(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:
    """
    Evaluates model predictions and returns the metrics.
    :param predicts: labels predicted by model
    :param target: actual target labels
    :return: a dict of metrics in format {'metric_name': value}
    """
    return {
        "accuracy": accuracy_score(target, predicts),
        "f1_score": f1_score(target, predicts),
        "roc_auc": roc_auc_score(target, predicts),
    }


@click.command("validate")
@click.option("--input_dir", required=True)
@click.option("--model_dir", required=True)
def validate(input_dir: str, model_dir: str):
    """
    Validates trained model.
    :input_dir: path to validation data
    :model_dir: path to trained model
    :return: nothing
    """
    logger.info('Starting model validation.')
    val_df = pd.read_csv(os.path.join(input_dir, VAL_FILENAME))

    logger.info("Loading model...")
    model = load_model(os.path.join(model_dir, MODEL_FILENAME))

    logger.info("Loading validation dataset...")
    features = val_df.drop(columns={TARGET_COL})
    target = val_df[TARGET_COL]
    predictions = model.predict(features)

    logger.info("Evaluating model...")
    metrics = evaluate_model(predictions, target)
    logger.info(f"Metrics are {metrics}.")

    logger.info("Saving metrics...")
    with open(os.path.join(model_dir, METRIC_FILENAME), "w") as metric_file:
        json.dump(metrics, metric_file)
        logger.info("All the metrics were successfully saved.")


if __name__ == '__main__':
    validate()
