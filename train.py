import logging
import joblib

from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import VotingRegressor
import numpy as np
import pandas as pd
import tensorflow as tf

from imbd.trainers import ModelTrainer
from imbd.data import DataLoader
from imbd.preprocessors import DataPreprocessor
from imbd.models import KerasModel
from imbd.inspectors import RegressionReport


def get_logger():
    logger = logging.getLogger(name='imbd2020')
    stream_handler = logging.StreamHandler()
    fmt = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)
    return logger


def main():
    logger = get_logger()
    logger.info("Start Training.")
    base_model = VotingRegressor([('xgb', XGBRegressor()),
                                  ('xgb_rf', XGBRFRegressor())])
    multi_output_model = MultiOutputRegressor(base_model)
    param_grid = {
        "prepro__variance_selector__threshold": [0.0, 0.01],
        # "voting__estimator__xgb__subsample": [1, 0.5],
        # "voting__estimator__xgb__max_depth": [2, 6],
        # "voting__estimator__xgb_rf__max_depth": [2, 6],
        # "voting__estimator__xgb_rf__subsample": [1, 0.5],
        "voting__estimator__xgb__n_estimators": [1000],
        "voting__estimator__xgb_rf__n_estimators": [1000],
    }

    # initialization
    loader = DataLoader()
    prepro = DataPreprocessor()
    df = loader.build()

    # get feature & label
    train_features = df.drop(loader.labels, axis=1)
    train_labels = df[loader.labels]

    # build pipeline
    steps = [('prepro', prepro), ('voting', multi_output_model)]
    pipe = Pipeline(steps=steps)

    # training
    trainer = ModelTrainer(pipe=pipe, param_grid=param_grid, verbose=2)
    logger.info("Start GridSearch.")
    fitted = trainer.fit(train_features, train_labels)
    report = RegressionReport(fitted)

    report.print_report()
    report.to_csv('models/cv_results.csv', index=False)
    logger.info("Save model.")
    joblib.dump(fitted, 'models/model.pkl')

    logger.info("Training finished.")


if __name__ == '__main__':
    main()