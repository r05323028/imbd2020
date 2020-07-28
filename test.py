import joblib
from argparse import ArgumentParser

import pandas as pd

from imbd.data import DataLoader
from imbd.utils import get_logger

TEAM_NUMBER = 109911


def get_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--file_path',
                            default='data/0728test.csv',
                            help='Data file path.',
                            type=str)
    arg_parser.add_argument('--model_path',
                            default='models/model.pkl',
                            help='Model file path.',
                            type=str)
    return arg_parser.parse_args()


def main(args):
    logger = get_logger()

    logger.info('Start Testing.')

    with open(args.model_path, 'rb') as file:
        model = joblib.load(file)

    loader = DataLoader(args.file_path, data_type='test')
    test_features = loader.build()
    pred = model.predict(test_features)
    pred = pd.DataFrame(pred, columns=loader.labels)

    # write into excel file
    logger.info("Write files...")
    pred.index = pred.index + 1
    pred.to_excel(f'results/{TEAM_NUMBER}_TestResult.xlsx', index_label='預測筆數')

    logger.info('Finished.')


if __name__ == '__main__':
    args = get_args()
    main(args)