# -*- coding: utf-8 -*-
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

from src.entities import SplittingParams


def read_data(path: str) -> pd.DataFrame:
    """
    Reads data from .csv file.
    :param path: path to .csv file to load from
    :return: pandas dataframe
    """
    data = pd.read_csv(path)
    return data


def split_train_val_data(
    data: pd.DataFrame, params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits data into training and validation datasets.
    :param data: initial dataframe
    :param params: splitting parameters
    :return: tuple of training and validation dataframes
    """
    train_data, val_data = train_test_split(
        data, test_size=params.val_size, random_state=params.random_state
    )
    return train_data, val_data
