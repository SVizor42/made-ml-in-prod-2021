from pydantic import BaseModel, conlist, validator
from typing import List, Union


FEATURES_MIN_MAX = [
    ("age", None, None),
    ("sex", 0, 1),
    ("cp", 0, 3),
    ("trestbps", None, None),
    ("chol", None, None),
    ("fbs", 0, 1),
    ("restecg", 0, 2),
    ("thalach", None, None),
    ("exang", 0, 1),
    ("oldpeak", None, None),
    ("slope", 0, 2),
    ("ca", 0, 4),
    ("thal", 0, 3),
]
FEATURE_NAMES = [ft for (ft, _, _) in FEATURES_MIN_MAX]
N_FEATURES = len(FEATURE_NAMES)

MAIN_ENDPOINT_MSG = "This is the entry point of our predictor."
OUT_OF_RANGE_MSG = "Feature value is out of range."
INCORRECT_FEATURES_MSG = "Incorrect features order or columns"


class HeartDiseaseModel(BaseModel):
    data: List[conlist(
        Union[int, float],
        min_items=N_FEATURES,
        max_items=N_FEATURES
    )]
    features: List[str]

    @validator("data")
    def check_data_limits(cls, data):
        for sample in data:
            for val, (_, min_val, max_val) in zip(sample, FEATURES_MIN_MAX):
                if (min_val is None) and (max_val is None):
                    continue
                if not (min_val <= val <= max_val):
                    raise ValueError(OUT_OF_RANGE_MSG)

        return data

    @validator("features")
    def check_features(cls, features):
        if features != FEATURE_NAMES:
            raise ValueError(INCORRECT_FEATURES_MSG)

        return features


class HeartDiseaseResponse(BaseModel):
    id: int
    disease: int
