import numpy as np
from collections.abc import Iterable
from lightgbm import LGBMRegressor


class BRT(LGBMRegressor):
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = -1,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs,
        )

    def batch_predict(self, iterator: Iterable[np.ndarray]) -> np.ndarray:
        predictions = []

        for batch in iterator:
            predictions.extend(self.predict(batch))

        return np.array(predictions)
