import logging
import os
import sys
import pandas as pd
import pickle
import click
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

TRAIN_FILENAME = "train.csv"
MODEL_FILENAME = "model.pkl"
TARGET_COL = "target"


@click.command("train")
@click.option("--input_dir", required=True)
@click.option("--model_dir", required=True)
@click.option("--random_state", default=42)
def train(input_dir: str, model_dir: str, random_state: int):
    """
    Trains the model.
    :input_dir: path to train data
    :model_dir: path to save training artifacts
    :random_state: model random state
    :return: nothing
    """
    logger.info('Starting model training.')
    logger.info("Loading training dataset...")
    train_df = pd.read_csv(os.path.join(input_dir, TRAIN_FILENAME))
    features = train_df.drop(columns={TARGET_COL})
    target = train_df[TARGET_COL]

    logger.info("Training model...")
    model = RandomForestClassifier(random_state=random_state)
    model.fit(features, target)

    logger.info("Saving model...")
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, MODEL_FILENAME), "wb") as file:
        pickle.dump(model, file)
        logger.info("The model was successfully trained and saved.")


if __name__ == '__main__':
    train()
