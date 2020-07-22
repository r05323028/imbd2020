import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV


class ModelTrainer(BaseEstimator):
    def __init__(self, pipe, param_grid, verbose=0, cv=3):
        self.scorer = make_scorer(mean_squared_error,
                                  squared=True,
                                  greater_is_better=False)
        self.model = GridSearchCV(pipe,
                                  param_grid,
                                  cv=cv,
                                  verbose=verbose,
                                  return_train_score=True,
                                  scoring='neg_root_mean_squared_error')

    @property
    def training_result(self):
        return pd.DataFrame(self.model.cv_results_)

    def fit(self, X, y=None, **fit_params):
        self.model.fit(X, y, **fit_params)

        return self.model

    def predict(self, X):
        return self.model.predict(X)
