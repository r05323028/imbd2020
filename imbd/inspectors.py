import plotly.express as px
import pandas as pd


class ResultInspector:
    def __init__(self, train_features, train_labels, predictions):
        self.train_features = train_features
        self.train_labels = train_labels
        self.predictions = predictions


class RegressionReport:
    def __init__(self, grid_search):
        self.model = grid_search

    @property
    def cv_result(self):
        return self.model.trials_dataframe()

    def print_report(self):
        print(f'Best Error (Test CV): {self.model.best_score_}')
        print(f'Best Params: {self.model.best_params_}')
        print(f'CV Result:\n{self.cv_result}')

    def to_csv(self, *args, **kwargs):
        self.cv_result.to_csv(*args, **kwargs)