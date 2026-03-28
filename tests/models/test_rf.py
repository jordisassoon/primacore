import pytest
import numpy as np
from primacore.models.rf import RF


@pytest.fixture
def sample_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate sample training and test data."""
    np.random.seed(42)
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100)
    X_test = np.random.rand(20, 10)
    return X_train, y_train, X_test


def test_initialization_custom_params() -> None:
    """Test RF initialization with custom parameters."""
    rf = RF(n_estimators=200, max_depth=5, random_state=123)

    assert rf.n_estimators == 200
    assert rf.max_depth == 5
    assert rf.random_state == 123


def test_batch_predict(sample_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Test batch_predict method."""
    X_train, y_train, X_test = sample_data

    rf = RF(n_estimators=200, max_depth=5, random_state=123)
    rf.fit(X_train, y_train)

    iterator = [X_test[i : i + 5] for i in range(0, X_test.shape[0], 5)]
    predictions = rf.batch_predict(iterator)

    assert predictions.shape == (20,)
    assert isinstance(predictions, np.ndarray)
