import pickle
import numpy as np
import pandas as pd
from typing import Dict, Union, Optional
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.entities.train_params import TrainingParams

ClassifierModel = Union[RandomForestClassifier, LogisticRegression]


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> ClassifierModel:
    """
    Trains the model.
    :param features: features to train on
    :param target: target labels to train on
    :param train_params: training parameters
    :return: trained classifier model
    """
    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            max_depth=train_params.max_depth,
            n_estimators=train_params.n_estimators,
            random_state=train_params.random_state,
        )
    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression(
            solver=train_params.solver,
            random_state=train_params.random_state,
        )
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


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


def serialize_model(
    model: ClassifierModel, output: str, transformer: Optional[ColumnTransformer] = None
) -> str:
    """
    Saves trained model (pipeline) to pickle file.
    :param transformer: data transformer to save
    :param model: trained model to save
    :param output: filename to save to
    :return: the path to pickle file
    """
    pipeline = Pipeline(
        [
            ('transformer', transformer),
            ('model', model),
        ]
    )
    with open(output, "wb") as file:
        pickle.dump(pipeline, file)
    return output
