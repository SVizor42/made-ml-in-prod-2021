import logging
import os
import sys
import pandas as pd
import pickle
import click

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

DATA_FILENAME = "data.csv"
TRANSFORMER_FILENAME = "transformer.pkl"
MODEL_FILENAME = "model.pkl"
PREDICTION_FILENAME = "predictions.csv"


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


@click.command("predict")
@click.option("--input_dir")
@click.option("--model_dir")
@click.option("--output_dir")
def predict(input_dir: str, model_dir: str, output_dir: str):
    """
    Makes predictions using model on given data.
    :input_dir: path to the features to predict on
    :model_dir: the model to predict with
    :output_dir: path to model predictions
    :return: nothing
    """
    logger.info('Starting to make predictions.')
    logger.info("Loading inference dataset and model...")
    features = pd.read_csv(os.path.join(input_dir, DATA_FILENAME))
    transformer = load_model(os.path.join(model_dir, TRANSFORMER_FILENAME))
    model = load_model(os.path.join(model_dir, MODEL_FILENAME))

    logger.info("Implementing transformations...")
    features_processed = pd.DataFrame(
        transformer.fit_transform(features), columns=features.columns
    )
    logger.info("Predicting...")
    predictions = pd.DataFrame(model.predict(features_processed))

    logger.info("Saving predictions...")
    os.makedirs(output_dir, exist_ok=True)
    try:
        predictions.to_csv(os.path.join(output_dir, PREDICTION_FILENAME), index=False)
        logger.info(f"Predictions were successfully saved to '{PREDICTION_FILENAME}' file.")
    except PermissionError as error:
        logger.error(f"Could not save the file due to the permission error. {error}")


if __name__ == '__main__':
    predict()
