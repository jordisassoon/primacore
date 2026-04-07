import numpy as np
from collections.abc import Iterable
from sklearn.ensemble import RandomForestRegressor


class RF(RandomForestRegressor):
    n_estimators: int
    max_depth: int | None
    random_state: int | None

    def __init__(self, n_estimators=100, max_depth=None, random_state=None, **kwargs):
        # Initialize the parent RandomForestRegressor
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs,
        )

    def batch_predict(self, iterator: Iterable[np.ndarray]) -> np.ndarray:
        predictions = []

        for batch in iterator:
            predictions.extend(self.predict(batch))

        return np.array(predictions)
