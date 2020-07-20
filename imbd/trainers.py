import pandas as pd
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV


class ModelTrainer:
    def __init__(self, pipe, param_grid, verbose=0, cv=3):
        self.scorer = make_scorer(mean_squared_error, squared=False)
        self.model = GridSearchCV(pipe,
                                  param_grid,
                                  cv=cv,
                                  verbose=verbose,
                                  scoring=self.scorer)

    @property
    def training_result(self):
        return pd.DataFrame(self.model.cv_results_)

    def train(self, features, labels):
        self.model.fit(features, labels)

        return self.model

    def predict(self, features):
        return self.model.predict(features)
