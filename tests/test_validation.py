import pytest
import numpy as np
import pandas as pd

from primacore.validation import (
    spearman_rho,
    pearson_r,
    kge,
    bias,
    build_scorers,
    spatial_cross_validation,
)
from primacore.models.rf import RF


# ── Scorer tests ──────────────────────────────────────────────────────────────


def test_spearman_rho_perfect_correlation() -> None:
    y_true = pd.Series([1, 2, 3, 4, 5])
    y_pred = pd.Series([1, 2, 3, 4, 5])
    assert spearman_rho(y_true, y_pred) == pytest.approx(1.0)


def test_spearman_rho_returns_zero_for_nan() -> None:
    y_true = pd.Series([1.0, 1.0, 1.0])
    y_pred = pd.Series([1.0, 1.0, 1.0])
    assert spearman_rho(y_true, y_pred) == 0.0


def test_pearson_r_perfect_correlation() -> None:
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0])
    assert pearson_r(y_true, y_pred) == pytest.approx(1.0)


def test_pearson_r_returns_zero_for_nan() -> None:
    y_true = pd.Series([1.0, 1.0, 1.0])
    y_pred = pd.Series([1.0, 1.0, 1.0])
    assert pearson_r(y_true, y_pred) == 0.0


def test_kge_perfect_prediction() -> None:
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert kge(y_true, y_pred) == pytest.approx(1.0)


def test_kge_handles_zero_std_and_mean() -> None:
    y_true = pd.Series([0.0, 0.0, 0.0])
    y_pred = pd.Series([1.0, 2.0, 3.0])
    result = kge(y_true, y_pred)
    assert np.isfinite(result)


def test_bias_no_bias() -> None:
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = pd.Series([1.0, 2.0, 3.0])
    assert bias(y_true, y_pred) == pytest.approx(0.0)


def test_bias_positive() -> None:
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = pd.Series([2.0, 3.0, 4.0])
    assert bias(y_true, y_pred) == pytest.approx(1.0)


def test_build_scorers_returns_requested_keys() -> None:
    scorers = build_scorers(["rmse", "r2", "pearson_r"])
    assert set(scorers.keys()) == {"rmse", "r2", "pearson_r"}


# ── Spatial cross-validation tests ────────────────────────────────────────────


@pytest.fixture
def cv_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.RandomState(42)
    n = 100
    X = pd.DataFrame(rng.rand(n, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.rand(n))
    groups = pd.Series(np.repeat(np.arange(5), n // 5))
    return X, y, groups


def test_spatial_cv_returns_dataframe(cv_data) -> None:
    X, y, groups = cv_data
    model = RF(n_estimators=10, random_state=0)
    scoring = build_scorers(["rmse", "r2"])

    result = spatial_cross_validation(model, X, y, groups, scoring, n_folds=5)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"rmse", "r2"}
    assert len(result) == 5


def test_spatial_cv_too_few_groups(cv_data) -> None:
    X, y, _ = cv_data
    groups = pd.Series(np.repeat([0, 1], len(y) // 2))
    model = RF(n_estimators=10, random_state=0)
    scoring = build_scorers(["rmse"])

    with pytest.raises(ValueError, match="Number of unique groups must be >= n_folds"):
        spatial_cross_validation(model, X, y, groups, scoring, n_folds=5)


def test_spatial_cv_custom_folds(cv_data) -> None:
    X, y, groups = cv_data
    model = RF(n_estimators=10, random_state=0)
    scoring = build_scorers(["mae"])

    result = spatial_cross_validation(model, X, y, groups, scoring, n_folds=3)

    assert len(result) == 3
    assert "mae" in result.columns


def test_spatial_cv_hardcoded_results(cv_data) -> None:
    X, y, groups = cv_data
    model = RF(n_estimators=10, random_state=0)
    scoring = build_scorers(["rmse", "r2"])

    result = spatial_cross_validation(model, X, y, groups, scoring, n_folds=5)

    expected_rmse = np.array(
        [-0.32980316, -0.25850395, -0.26847311, -0.35891507, -0.35157779]
    )
    expected_r2 = np.array(
        [-0.49778018, 0.12580646, -0.11568117, -0.46708625, -0.28533652]
    )

    np.testing.assert_allclose(result["rmse"].values, expected_rmse, rtol=1e-5)
    np.testing.assert_allclose(result["r2"].values, expected_r2, rtol=1e-5)
