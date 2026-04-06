from typing import Callable, Dict, Union

import numpy as np
import pandas as pd

from primacore.models.brt import BRT
from primacore.models.mat import MAT
from primacore.models.rf import RF

from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import make_scorer

from scipy.stats import pearsonr, spearmanr

Scoring = Union[str, Callable]


def spearman_rho(y_true: pd.Series, y_pred: pd.Series) -> float:
    rho, _ = spearmanr(y_true, y_pred)
    return 0.0 if np.isnan(rho) else rho


def pearson_r(y_true: pd.Series, y_pred: pd.Series) -> float:
    r, _ = pearsonr(y_true, y_pred)
    return 0.0 if np.isnan(r) else r


def kge(y_true: pd.Series, y_pred: pd.Series) -> float:
    r = pearson_r(y_true, y_pred)

    std_true = np.std(y_true)
    mean_true = np.mean(y_true)

    alpha = np.std(y_pred) / std_true if std_true != 0 else 0.0
    beta = np.mean(y_pred) / mean_true if mean_true != 0 else 0.0

    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def bias(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(y_pred - y_true))


def build_scorers(scorers_list: list[str] | None = None) -> Dict[str, Scoring]:
    """
    Create a sklearn-compatible scoring dictionary.
    """
    all_scorers = {
        "rmse": "neg_root_mean_squared_error",
        "mse": "neg_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
        "pearson_r": make_scorer(pearson_r),
        "spearman_rho": make_scorer(spearman_rho),
        "kge": make_scorer(kge),
        "bias": make_scorer(bias, greater_is_better=False),
    }

    if scorers_list is None:
        return all_scorers

    return {key: all_scorers[key] for key in scorers_list}


def spatial_cross_validation(
    model: Union[RF, BRT, MAT],
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    scoring: Dict[str, Scoring],
    n_folds: int = 5,
) -> pd.DataFrame:
    """
    Perform grouped spatial cross-validation using sklearn.
    """

    if groups.nunique() < n_folds:
        raise ValueError("Number of unique groups must be >= n_folds")

    cv = GroupKFold(n_splits=n_folds)

    results = cross_validate(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    df = pd.DataFrame(results)

    df = df.filter(like="test_")
    df.columns = df.columns.str.replace("test_", "")

    return df
