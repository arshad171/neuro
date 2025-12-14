from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np

batch_size = 16
n_features = 784
max_depth = 10

def get_data(batch_size):
    data = np.random.uniform(size=(batch_size, n_features))

    return data


class Model():
    def __init__(self, device_type=None):
        X, y = make_classification(n_samples=10_000, n_features=n_features,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
        self.model = RandomForestClassifier(max_depth=max_depth, random_state=0)
        self.model.fit(X, y)

    def predict(self, batch_size=batch_size):
        data = get_data(batch_size)

        preds = self.model.predict(data)

        return preds.tolist()
    