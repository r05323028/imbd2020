import pytest
from imbd.data import DataLoader, DataPreprocessor


def test_data():
    loader = DataLoader()
    df = loader.build_label_20_df()
    preprocessor = DataPreprocessor()
    res = preprocessor(df)
