from argparse import ArgumentParser
import joblib

from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.pipeline import Pipeline
from sklearn.multioutput import RegressorChain
from sklearn.ensemble import VotingRegressor
import numpy as np
import pandas as pd
import tensorflow as tf

from imbd.trainers import ModelTrainer
from imbd.data import DataLoader
from imbd.preprocessors import DataPreprocessor
from imbd.models import KerasModel
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
    logger.info("Start Training.")
    base_model = VotingRegressor([('xgb', XGBRegressor()),
                                  ('xgb_rf', XGBRFRegressor())])
    order = [0, 2, 5, 7, 13, 14, 16, 17
             ] + [1, 3, 4, 6, 8, 9, 11, 12, 15, 18, 19] + [10]
    multi_output_model = RegressorChain(base_model, order=order)
    param_grid = {}

    # param_grid = {
    #     "prepro__variance_selector__threshold": [0.0],
    #     "prepro__cluster_maker__n_cluster": [10, 15],
    #     # "prepro__pca_embedder__n_comp": [2, 5],
    #     "voting__base_estimator__xgb__subsample": [1, 0.5],
    #     "voting__base_estimator__xgb__max_depth": [2, 6],
    #     "voting__base_estimator__xgb__colsample_bytree": [1, 0.5],
    #     "voting__base_estimator__xgb__colsample_bylevel": [1, 0.5],
    #     "voting__base_estimator__xgb__colsample_bynode": [1, 0.5],
    #     "voting__base_estimator__xgb_rf__max_depth": [2, 6],
    #     "voting__base_estimator__xgb_rf__subsample": [1, 0.5],
    #     "voting__base_estimator__weights": [[0.4, 0.6], [0.5, 0.5]],
    #     "voting__base_estimator__xgb__n_estimators": [1000, 10000],
    #     "voting__base_estimator__xgb_rf__n_estimators": [1000, 10000],
    # }

    # initialization
    loader = DataLoader(data_fp=args.file_path)
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
    args = get_args()
    main(args)