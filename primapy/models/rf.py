import numpy as np
from sklearn.ensemble import RandomForestRegressor


class RF(RandomForestRegressor):
    def __init__(self, n_estimators=100, max_depth=None, random_state=None, **kwargs):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        # Initialize the parent RandomForestRegressor
        super().__init__(
            n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state, **kwargs
        )

    def predict_with_progress(self, iterator):
        predictions = []

        for batch in iterator:
            predictions.extend(self.predict(batch))

        return np.array(predictions)