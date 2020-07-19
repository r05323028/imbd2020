import pandas as pd
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor


class ModelTrainer:
    cv_folds = 3

    def __init__(self, model=None, param_grid=None):
        self.scorer = make_scorer(mean_squared_error, squared=False)
        self.base_estimator = MultiOutputRegressor(XGBRegressor())
        if param_grid:
            self.param_grid = param_grid
        else:
            self.param_grid = {
                "estimator__n_estimators": [100, 1000],
                "estimator__max_depth": [2, 5, 10],
                "estimator__alpha": [0, 0.1, 0.01],
                "estimator__lambda": [1, 0.5, 0.1],
                "estimator__subsample": [1, 0.5, 0.1],
                "estimator__gamma": [0, 2, 10],
            }
        if model:
            self.model = model
        else:
            self.model = GridSearchCV(self.base_estimator,
                                      self.param_grid,
                                      cv=self.cv_folds,
                                      verbose=2,
                                      scoring=self.scorer)

    @property
    def training_result(self):
        return pd.DataFrame(self.model.cv_results_)

    def train(self, features, labels):
        self.model.fit(features, labels)

        return self.model

    def predict(self, features):
        return self.model.predict(features)
