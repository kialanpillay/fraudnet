from scipy.stats import stats


class NaiveClassifier:
    def __init__(self):
        self.y_pred = None

    def fit(self, X, y):
        self.y_pred = stats.mode(y)[0][0]

    def predict(self, X):
        return np.full(shape=X.shape[0], fill_value=self.y_pred, dtype=np.float32)

    def fit_predict(self, X, y):
        self.fit(X, y)
        self.predict(X)