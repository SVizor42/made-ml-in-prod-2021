import logging
import os
import sys
import numpy as np
import pandas as pd
import click
from typing import Tuple

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

DATA_FILENAME = "data.csv"
TARGET_FILENAME = "target.csv"


@click.command("generate")
@click.option("--output_dir")
def generate(output_dir: str):
    logger.info('Starting to generate raw data.')
    logger.info('Generating dataset...')
    features, target = generate_dataset(size=100)

    logger.info("Saving generated data...")
    os.makedirs(output_dir, exist_ok=True)
    try:
        features.to_csv(os.path.join(output_dir, DATA_FILENAME), index=False)
        logger.info(f"File '{DATA_FILENAME}' was successfully saved.")
        target.to_csv(os.path.join(output_dir, TARGET_FILENAME), index=False, header=True)
        logger.info(f"File '{TARGET_FILENAME}' was successfully saved.")
    except PermissionError as error:
        logger.error(f"Could not save the file due to the permission error. {error}")


def generate_dataset(size: int = 50, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(seed)
    data = pd.DataFrame()
    data["age"] = np.random.normal(loc=54, scale=9, size=size).astype(int)
    data["sex"] = np.random.binomial(n=1, p=0.7, size=size).astype(int)
    data["cp"] = np.random.randint(low=0, high=4, size=size).astype(int)
    data["trestbps"] = np.random.normal(loc=131, scale=18, size=size).astype(int)
    data["chol"] = np.random.normal(loc=246, scale=52, size=size).astype(int)
    data["fbs"] = np.random.binomial(n=1, p=0.15, size=size).astype(int)
    data["restecg"] = np.random.randint(low=0, high=3, size=size).astype(int)
    data["thalach"] = np.random.normal(loc=150, scale=23, size=size).astype(int)
    data["exang"] = np.random.binomial(n=1, p=0.33, size=size).astype(int)
    data["oldpeak"] = np.clip(np.random.normal(loc=1, scale=2, size=size), 0, None).astype(int)
    data["slope"] = np.random.randint(low=0, high=3, size=size).astype(int)
    data["ca"] = np.random.randint(low=0, high=5, size=size).astype(int)
    data["thal"] = np.random.randint(low=0, high=4, size=size).astype(int)
    target = pd.DataFrame(columns=["target"])
    target["target"] = np.random.binomial(n=1, p=0.55, size=size).astype(int)
    return data, target


if __name__ == '__main__':
    generate()
