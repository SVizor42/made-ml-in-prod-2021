import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.entities.feature_params import FeatureParams


def process_categorical_features(categorical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds pipeline and processes categorical features.
    :param categorical_df: dataframe of categorical features
    :return: dataframe of the processed features
    """
    categorical_pipeline = build_categorical_pipeline()
    return pd.DataFrame(categorical_pipeline.fit_transform(categorical_df).toarray())


def build_categorical_pipeline() -> Pipeline:
    """
    Builds pipeline for categorical features processing.
    :return: pipeline class
    """
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return categorical_pipeline


def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds pipeline and processes numerical features.
    :param numerical_df: dataframe of numerical features
    :return: dataframe of the processed features
    """
    num_pipeline = build_numerical_pipeline()
    return pd.DataFrame(num_pipeline.fit_transform(numerical_df))


def build_numerical_pipeline() -> Pipeline:
    """
    Builds pipeline for numerical features processing.
    :return: pipeline class
    """
    num_pipeline = Pipeline(
        [("impute", SimpleImputer(missing_values=np.nan, strategy="mean")), ]
    )
    return num_pipeline


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    """
    Builds transformer to process both categorical and numerical features.
    :param params: configuration for feature processing
    :return: transformer class
    """
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                params.categorical_features,
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                params.numerical_features,
            ),
        ]
    )
    return transformer


def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes features.
    :param transformer: transformer to process features
    :param df: features dataframe
    :return: dataframe of the processed features
    """
    return pd.DataFrame(transformer.transform(df))


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    """
    Extracts target from dataframe.
    :param df: features dataframe
    :param params: configuration for feature processing
    :return: series of the processed target
    """
    target = df[params.target_col] if params.target_col in df.columns else None
    return target
