import pytest
import numpy as np
from primapy.models.brt import BRT

@pytest.fixture
def sample_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate sample training and test data."""
    np.random.seed(42)
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100)
    X_test = np.random.rand(20, 10)
    return X_train, y_train, X_test

def test_initialization_custom_params() -> None:
    """Test BRT initialization with custom parameters."""
    brt = BRT(n_estimators=200, verbose=-1)

    assert brt.n_estimators == 200 # Exposed class attribute
    assert brt.verbose == -1 # Override with **kwargs
    assert brt.learning_rate == 0.1  # Default value

def test_predict_with_progress(sample_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Test predict_with_progress method."""
    X_train, y_train, X_test = sample_data

    brt = BRT(random_state=42, verbose=-1)
    brt.fit(X_train, y_train)

    iterator = [X_test[i:i+5] for i in range(0, X_test.shape[0], 5)]
    predictions = brt.batch_predict(X_test, iterator)

    assert predictions.shape == (20,)
    assert isinstance(predictions, np.ndarray)