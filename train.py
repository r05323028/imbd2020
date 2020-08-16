from argparse import ArgumentParser
import joblib

from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.multioutput import RegressorChain
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import tensorflow as tf
import optuna

from imbd.trainers import OptunaModelTrainer
from imbd.data import DataLoader
from imbd.preprocessors import DataPreprocessor
from imbd.inspectors import RegressionReport
from imbd.utils import get_logger


def get_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--file_path',
                            default='data/0714train.csv',
                            help='Data file path.',
                            type=str)
    return arg_parser.parse_args()


def main(args):
    logger = get_logger()
    base_model = LGBMRegressor()
    logger.info("Start Training.")
    order = [0, 2, 5, 7, 13, 14, 16, 17
             ] + [1, 3, 4, 6, 8, 9, 11, 12, 15, 18, 19] + [10]
    multi_output_model = RegressorChain(base_model, order=order)

    param_grid = {
        "base_estimator__max_depth":
        optuna.distributions.IntLogUniformDistribution(1, 20),
        "base_estimator__min_child_samples":
        optuna.distributions.IntLogUniformDistribution(1, 20),
        "base_estimator__subsample":
        optuna.distributions.LogUniformDistribution(0.5, 1),
        "base_estimator__n_estimators":
        optuna.distributions.IntLogUniformDistribution(1000, 10000),
        "base_estimator__colsample_bytree":
        optuna.distributions.LogUniformDistribution(0.5, 1),
        "base_estimator__num_leaves":
        optuna.distributions.IntLogUniformDistribution(15, 90),
        "base_estimator__reg_alpha":
        optuna.distributions.LogUniformDistribution(0.001, 0.01),
        "base_estimator__reg_lambda":
        optuna.distributions.LogUniformDistribution(0.001, 0.01),
        "base_estimator__feature_fraction":
        optuna.distributions.LogUniformDistribution(0.5, 1),
        "base_estimator__colsample_bynode":
        optuna.distributions.LogUniformDistribution(0.5, 1),
        "base_estimator__bagging_fraction":
        optuna.distributions.LogUniformDistribution(0.5, 1),
        "base_estimator__min_data_in_leaf":
        optuna.distributions.IntLogUniformDistribution(1, 20),
    }

    # initialization
    loader = DataLoader(data_fp=args.file_path)
    prepro = DataPreprocessor()
    df = loader.build()

    # get feature & label
    train_features = df.drop(loader.labels, axis=1)
    train_labels = df[loader.labels]

    train_features = prepro.fit_transform(train_features, train_labels)

    # training
    trainer = OptunaModelTrainer(base_model=multi_output_model,
                                 param_grid=param_grid,
                                 verbose=2,
                                 n_jobs=-1)
    logger.info("Start OptunaSearchCV.")
    fitted = trainer.fit(train_features, train_labels)
    report = RegressionReport(fitted)

    report.print_report()
    report.to_csv('models/cv_results.csv', index=False)
    logger.info("Save model.")
    joblib.dump(prepro, "models/preprocessor.pkl")
    joblib.dump(fitted, 'models/model.pkl')

    logger.info("Training finished.")


if __name__ == '__main__':
    args = get_args()
    main(args)