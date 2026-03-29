from typing import Union
import pandas as pd
from primacore.models.brt import BRT
from primacore.models.mat import MAT
from primacore.models.rf import RF

from sklearn.model_selection import cross_val_score, GroupKFold


def spatial_cross_validation(
    model: Union[BRT, MAT, RF, None],
    X: pd.DataFrame,
    y: pd.DataFrame,
    groups: pd.Series,
    n_folds: int = 5,
    scoring: str = "neg_mean_squared_error",
) -> tuple:
    """
    Perform cross-validation on a fitted model while preventing data leakage.

    Groups data by observation name to ensure same obs_name stays in same fold.
    """
    group_kfold = GroupKFold(n_splits=n_folds)

    scores = cross_val_score(
        model,
        X,
        y,
        cv=group_kfold,
        groups=groups,
        scoring=scoring,
    )

    mean_score = scores.mean()
    std_score = scores.std()

    return scores, mean_score, std_score
