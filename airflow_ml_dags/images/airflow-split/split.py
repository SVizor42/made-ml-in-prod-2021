import logging
import os
import sys
import pandas as pd
import click
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

PROCESSED_FILENAME = "processed.csv"
TRAIN_FILENAME = "train.csv"
VAL_FILENAME = "val.csv"


@click.command("split")
@click.option("--input_dir", required=True)
@click.option("--output_dir", required=True)
@click.option("--train_size", default=0.8)
@click.option("--random_state", default=42)
def split(input_dir: str, output_dir: str, train_size: float, random_state: int):
    """
    Splits dataset into train and validation subsets.
    :input_dir: path to initial data
    :output_dir: path to splitted data
    :train_size: proportion of the dataset to include in the train part
    :random_state: controls the shuffling applied to the data before splitting
    :return: nothing
    """
    logger.info('Starting dataset splitting.')
    data_processed = pd.read_csv(os.path.join(input_dir, PROCESSED_FILENAME))

    logger.info("Splitting dataset...")
    train, val = train_test_split(data_processed,
                                  train_size=train_size,
                                  random_state=random_state)

    logger.info("Saving train and validation datasets...")
    os.makedirs(output_dir, exist_ok=True)
    try:
        train.to_csv(os.path.join(output_dir, TRAIN_FILENAME), index=False)
        logger.info(f"Train dataset was successfully saved to '{TRAIN_FILENAME}' file.")
        val.to_csv(os.path.join(output_dir, VAL_FILENAME), index=False)
        logger.info(f"Validation dataset was successfully saved to '{VAL_FILENAME}' file.")
        logger.info("Data were successfully splitted into train and validation parts.")
    except PermissionError as error:
        logger.error(f"Could not save the file due to the permission error. {error}")


if __name__ == '__main__':
    split()
