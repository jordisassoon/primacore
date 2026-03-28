import numpy as np
from collections.abc import Callable, Iterable
from sklearn.neighbors import KNeighborsRegressor


def squared_chord_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    return np.sum((np.sqrt(x1) - np.sqrt(x2)) ** 2)


def chord_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    return np.sqrt(np.sum((np.sqrt(x1) - np.sqrt(x2)) ** 2))


class MAT(KNeighborsRegressor):
    def __init__(
        self,
        n_neighbors: int = 3,
        metric: Callable[[np.ndarray, np.ndarray], float] = squared_chord_distance,
        **kwargs,
    ) -> None:
        if metric is not squared_chord_distance and metric is not chord_distance:
            raise ValueError("Invalid distance_metric. Choose squared_chord or chord.")
        super().__init__(n_neighbors=n_neighbors, metric=metric, **kwargs)

    def batch_predict(self, iterator: Iterable[np.ndarray]) -> np.ndarray:
        predictions = []

        for batch in iterator:
            predictions.extend(self.predict(batch))

        return np.array(predictions)
