import numpy as np
import pandas as pd
from typing import NoReturn
from sklearn.base import BaseEstimator, TransformerMixin


class CustomSqrtTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for specific numerical feature.
    """
    def __init__(self, feature_name: str, copy: bool = True) -> NoReturn:
        self.feature_name = feature_name
        super(__class__, self).__init__()

    def fit(self, x: pd.DataFrame, y: pd.Series = None) -> "CustomSqrtTransformer":
        """
        Fits data.
        :param x: input data to fit
        :param y: target labels
        :return: fitted data
        """
        return self

    def transform(self, x: pd.DataFrame, y: pd.Series = None) -> pd.Series:
        """
        Transforms data.
        :param x: data to transform
        :param y: target labels
        :return: transformed data
        """
        x_copy = x.copy()
        x_copy[self.feature_name] = np.sqrt(x_copy[self.feature_name])
        return x_copy

    def fit_transform(self,  x: pd.DataFrame, y: pd.Series = None) -> pd.Series:
        """
        Fits and transforms data.
        :param x: data to transform
        :param y: target labels
        :return: transformed data
        """
        return self.fit(x, y).transform(x, y)
