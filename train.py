from argparse import ArgumentParser
import joblib
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.multioutput import RegressorChain
import numpy as np
import pandas as pd
import tensorflow as tf

from imbd.trainers import ModelTrainer
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
    param_grid = {}

    param_grid = {
        "base_estimator__n_estimators": [1000, 10000],
        "base_estimator__boosting_type": ['gbdt', 'dart', 'rf'],
        # "base_estimator__tree_learner":
        # ["serial", "data", "feature", "voting"],
        # "base_estimator__max_depth": [2, 6, -1],
        "base_estimator__min_child_samples": [10, 20],
        "base_estimator__subsample": [0.5, 1],
        # "base_estimator__num_leaves ": [15, 31],
        "base_estimator__colsample_bytree": [0.5, 1],
        # "base_estimator__reg_alpha": [0.0, 0.05],
        # "base_estimator__reg_lambda": [0.0, 0.05],
    }

    # initialization
    loader = DataLoader(data_fp=args.file_path)
    prepro = DataPreprocessor()
    df = loader.build()

    # get feature & label
    train_features = df.drop(loader.labels, axis=1)
    train_labels = df[loader.labels]

    train_features = prepro.fit_transform(train_features)

    # training
    trainer = ModelTrainer(base_model=multi_output_model,
                           param_grid=param_grid,
                           verbose=2)
    logger.info("Start GridSearch.")
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