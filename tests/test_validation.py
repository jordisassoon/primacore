import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from primacore.models.rf import RF
from primacore.validation import spatial_cross_validation


@pytest.fixture()
def regression_data():
    rng = np.random.default_rng(42)
    n_samples = 60
    n_features = 3
    n_groups = 6

    X = pd.DataFrame(
        rng.standard_normal((n_samples, n_features)), columns=["f1", "f2", "f3"]
    )
    y = pd.DataFrame(rng.standard_normal(n_samples), columns=["target"])
    groups = pd.Series(np.repeat(np.arange(n_groups), n_samples // n_groups))
    return X, y, groups


@pytest.fixture()
def model():
    return RF(n_estimators=10, random_state=0)


def test_returns_scores_mean_and_std(model, regression_data):
    """Return value is a tuple of (scores_array, mean, std)."""
    X, y, groups = regression_data
    scores, mean_score, std_score = spatial_cross_validation(model, X, y, groups)

    assert isinstance(scores, np.ndarray)
    assert isinstance(mean_score, float)
    assert isinstance(std_score, float)


def test_number_of_scores_matches_n_folds(model, regression_data):
    """Length of scores array equals the requested number of folds."""
    X, y, groups = regression_data

    for n_folds in (3, 5):
        scores, _, _ = spatial_cross_validation(model, X, y, groups, n_folds=n_folds)
        assert len(scores) == n_folds


def test_mean_and_std_are_consistent_with_scores(model, regression_data):
    """Reported mean and std match np.mean / np.std of the raw scores."""
    X, y, groups = regression_data
    scores, mean_score, std_score = spatial_cross_validation(model, X, y, groups)

    assert mean_score == pytest.approx(scores.mean())
    assert std_score == pytest.approx(scores.std())


def test_scoring_parameter_is_respected(model, regression_data):
    """Different scoring metrics produce different score values."""
    X, y, groups = regression_data

    _, mean_mse, _ = spatial_cross_validation(
        model, X, y, groups, scoring="neg_mean_squared_error"
    )
    _, mean_r2, _ = spatial_cross_validation(model, X, y, groups, scoring="r2")

    assert mean_mse != pytest.approx(mean_r2)


def test_groups_do_not_leak_across_folds(regression_data):
    """No spatial group appears in both train and test within a single fold."""
    X, y, groups = regression_data
    n_folds = 5
    gkf = GroupKFold(n_splits=n_folds)

    for train_idx, test_idx in gkf.split(X, y, groups):
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        assert train_groups.isdisjoint(test_groups)


def test_deterministic_with_fixed_seed(regression_data):
    """Same model seed and data produce identical scores across runs."""
    X, y, groups = regression_data

    scores_a, _, _ = spatial_cross_validation(
        RF(n_estimators=10, random_state=0), X, y, groups
    )
    scores_b, _, _ = spatial_cross_validation(
        RF(n_estimators=10, random_state=0), X, y, groups
    )

    np.testing.assert_array_equal(scores_a, scores_b)
