import logging
import os
import sys
import pandas as pd
import pickle
import click
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

DATA_FILENAME = "data.csv"
TARGET_FILENAME = "target.csv"
PROCESSED_FILENAME = "processed.csv"
TRANSFORMER_FILENAME = "transformer.pkl"


@click.command("preprocess")
@click.option("--input_dir")
@click.option("--output_dir")
@click.option("--model_dir")
def preprocess(input_dir: str, output_dir: str, model_dir: str):
    """
    Preprocesses train data.
    :input_dir: path to raw train data
    :output_dir: path to processed train data
    :model_dir: path to save data transformer model
    :return: nothing
    """
    logger.info('Starting data preprocessing.')
    features = pd.read_csv(os.path.join(input_dir, DATA_FILENAME))
    target = pd.read_csv(os.path.join(input_dir, TARGET_FILENAME))

    logger.info("Implementing transformations...")
    scaler = StandardScaler()
    features_processed = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    data_processed = pd.concat([features_processed, target], axis=1)

    logger.info("Saving processed dataset...")
    os.makedirs(output_dir, exist_ok=True)
    try:
        data_processed.to_csv(os.path.join(output_dir, PROCESSED_FILENAME), index=False)
        logger.info(f"Data were successfully processed and saved to '{PROCESSED_FILENAME}' file.")
    except PermissionError as error:
        logger.error(f"Could not save the file due to the permission error. {error}")

    logger.info("Saving data transformer...")
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, TRANSFORMER_FILENAME), "wb") as file:
        pickle.dump(scaler, file)
        logger.info("Data transformer was successfully saved.")


if __name__ == '__main__':
    preprocess()
