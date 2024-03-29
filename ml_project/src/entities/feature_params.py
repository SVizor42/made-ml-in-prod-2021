from dataclasses import dataclass
from typing import List, Optional


@dataclass()
class FeatureParams:
    """
    Dataclass for feature and target parameters configuration.
    """

    categorical_features: List[str]
    numerical_features: List[str]
    features_to_drop: List[str]
    target_col: Optional[str]
