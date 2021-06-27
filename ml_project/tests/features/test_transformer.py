import numpy as np
import pandas as pd
import pytest
from typing import List

from src.features.transformer import CustomSqrtTransformer
from src.features.build_features import process_numerical_features


@pytest.fixture(scope="function")
def numerical_feature() -> str:
    return "numerical_feature"


@pytest.fixture(scope="function")
def numerical_values() -> List[float]:
    return [0., 9., 16.1, 169., 45.3]


@pytest.fixture(scope="function")
def numerical_values_with_nan(numerical_values: List[float]) -> List[float]:
    return numerical_values + [float(np.nan)]


@pytest.fixture(scope="function")
def fake_numerical_data(
    numerical_feature: str, numerical_values_with_nan: List[float]
) -> pd.DataFrame:
    return pd.DataFrame({numerical_feature: numerical_values_with_nan})


def test_transformer(
    numerical_feature: str,
    fake_numerical_data: pd.DataFrame,
):
    transformed: pd.DataFrame = process_numerical_features(fake_numerical_data)
    transformed.columns = {numerical_feature}
    transformer = CustomSqrtTransformer(numerical_feature)
    result: pd.Series = transformer.fit_transform(transformed)
    expected_result = np.sqrt(transformed.to_numpy())
    assert 6 == result.shape[0]
    assert np.allclose(result.to_numpy(), expected_result)
