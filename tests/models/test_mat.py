import pytest
import pandas as pd
import numpy as np
from primacore.models.mat import MAT, chord_distance


@pytest.fixture
def sample_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate sample training and test data."""
    np.random.seed(42)
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100)
    X_test = np.random.rand(20, 10)
    return X_train, y_train, X_test


def test_initialization_custom_params() -> None:
    """Test MAT initialization with custom parameters."""
    mat = MAT(n_neighbors=5, metric=chord_distance)

    assert mat.n_neighbors == 5
    assert mat.metric == chord_distance


def test_batch_predict(sample_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Test batch_predict method."""
    X_train, y_train, X_test = sample_data

    mat = MAT(n_neighbors=5, metric=chord_distance)
    mat.fit(X_train, y_train)

    iterator = [X_test[i : i + 5] for i in range(0, X_test.shape[0], 5)]
    predictions = mat.batch_predict(iterator)

    assert predictions.shape == (20,)
    assert isinstance(predictions, np.ndarray)


def test_get_neighbors(sample_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Test get_neighbors method."""
    X_train, y_train, X_test = sample_data

    mat = MAT(n_neighbors=3, metric=chord_distance)
    mat.fit(X_train, y_train)

    neighbors_df = mat.get_neighbors(pd.DataFrame(X_test))

    assert isinstance(neighbors_df, pd.DataFrame)
    assert set(neighbors_df.columns) == {"sample", "neighbor", "distance"}
    assert len(neighbors_df) == 20 * 3  # 20 samples * 3 neighbors each
