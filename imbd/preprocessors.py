import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


class QuantizationTransformer(TransformerMixin):
    '''
    Transform cells into pandas categorical dtype.
    '''
    one_hot_cols = {
        'Input_A3_010': 2,
        'Input_A4_008': 2,
        'Input_A1_008': 2,
        'Input_A6_008': 3,
        'Input_A5_008': 3,
        'Input_A1_023': 3,
        'Input_A3_009': 4,
        'Input_A4_009': 4,
        'Input_A2_013': 4,
        'Input_A4_014': 4,
        'Input_A3_008': 4,
        'Input_A3_014': 4,
        'Input_A1_013': 4,
        'Input_A1_009': 4,
        'Input_A5_009': 4,
        'Input_A5_013': 4,
        'Input_A1_014': 4,
        'Input_A6_009': 5,
        'Input_A6_013': 5,
        'Input_A2_019': 5,
        'Input_A5_016': 5,
        'Input_A4_013': 5,
        'Input_A4_016': 5,
        'Input_A4_017': 5,
        'Input_A4_018': 5,
        'Input_A4_019': 5,
        'Input_A5_019': 5,
        'Input_A5_018': 5,
        'Input_A5_014': 5,
        'Input_A3_019': 5,
        'Input_A2_018': 5,
        'Input_A5_017': 5,
        'Input_A2_014': 5,
        'Input_A6_018': 5,
        'Input_A1_016': 5,
        'Input_A1_017': 5,
        'Input_A1_018': 5,
        'Input_A1_019': 5,
        'Input_A6_017': 5,
        'Input_A1_022': 5,
        'Input_A6_016': 5,
        'Input_A6_014': 5,
        'Input_A2_009': 5,
        'Input_A1_024': 6,
        'Input_A3_011': 7,
        'Input_A2_003': 7,
        'Input_A5_011': 8,
        'Input_A3_003': 8,
        'Input_A2_023': 8,
        'Input_A4_003': 8,
        'Input_A5_003': 8,
        'Input_A3_004': 9,
        'Input_A5_024': 9,
        'Input_A4_011': 9,
        'Input_A2_011': 9,
        'Input_A5_004': 9,
        'Input_A1_011': 9,
        'Input_A2_012': 9,
        'Input_A6_022': 9,
        'Input_A5_022': 9,
        'Input_A6_003': 9,
        'Input_A2_022': 9,
        'Input_A2_001': 9
    }

    def fit(self, X, y=None, **fit_params):

        return self

    def transform(self, X, y=None):
        df = X.copy()
        dfs = []
        temp_cols = {}
        intersec = list(set(X.columns) & set(self.one_hot_cols.keys()))

        for col in intersec:
            temp_cols[col] = self.one_hot_cols[col]

        target_df = X[temp_cols]
        target_df = target_df.astype('category')

        for col, depth in temp_cols.items():
            feature_names = [f'{col}_one_hot_{i}' for i in range(depth)]
            one_hot = tf.one_hot(target_df[col], depth=depth)
            one_hot_df = pd.DataFrame(one_hot.numpy(),
                                      columns=feature_names,
                                      index=target_df.index)
            dfs.append(one_hot_df)

        df_one_hot = pd.concat(dfs, axis=1)
        df_ret = pd.concat([df, df_one_hot], axis=1)
        # df_ret = df_ret.drop(intersec, axis=1)
        return df_ret


class NAAnnotationTransformer(TransformerMixin):
    '''
    Annotate whether Input_C_083 ~ Input_C_091 is na.
    '''
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        df['massive_missing'] = df[[
            f'Input_C_{col:03d}' for col in range(83, 92)
        ]].isna().sum(axis=1) > 0
        df['massive_missing'] = df['massive_missing'].astype('float')
        return df


class FillNATransformer(TransformerMixin):
    '''
    Filling na cells.
    
    rules:
        category -> mode
        float, int -> mean
    '''
    def fit(self, X, y=None, **fit_params):
        # knn imputers
        self.knn_imputer = KNNImputer()
        self.knn_imputer.fit(X)

        return self

    def transform(self, X, y=None):
        df = X.copy()

        # simple imputer
        # df[float_columns] = self.mean_imputer.fit_transform(X[float_columns])
        # df[category_columns] = self.mode_imputer.fit_transform(
        #     X[category_columns])

        # knn transform
        # df[self.float_columns] = self.float_knn_imputer.transform(
        #     X[self.float_columns])
        # df[self.category_columns] = self.category_knn_imputer.transform(
        #     X[self.category_columns])
        df = pd.DataFrame(self.knn_imputer.transform(X))
        df.columns = X.columns

        return df


class OutlierDetector(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        self.A020_columns = X.filter(regex='Input_A[0-9]_020').columns

        self.iforest = IsolationForest(n_estimators=1000)
        self.iforest.fit(X[self.A020_columns])

        return self

    def transform(self, X, y=None):
        df = X.copy()

        df['outlier'] = self.iforest.predict(X[self.A020_columns])

        return df


class ClusterTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, n_cluster: int = 3):
        self.n_cluster = n_cluster

    def set_params(self, **params):
        super(ClusterTransformer, self).set_params(**params)

    def fit(self, X, y=None, **fit_params):
        self.model = KMeans(self.n_cluster)
        self.model.fit(X)
        return self

    def transform(self, X, y=None):
        df = X.copy()
        clusters = self.model.predict(X)
        clusters = tf.one_hot(clusters, depth=self.n_cluster)
        clusters = pd.DataFrame(
            clusters.numpy(),
            columns=[f'cluster_{i}' for i in range(self.n_cluster)])
        clusters.index = df.index
        df = pd.concat([df, clusters], axis=1)

        return df


class VarianceFeatureSelector(TransformerMixin, BaseEstimator):
    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def set_params(self, **params):
        super(VarianceFeatureSelector, self).set_params(**params)

    def fit(self, X, y=None, **fit_params):
        self.selector = VarianceThreshold(self.threshold)
        self.selector.fit(X)

        return self

    def transform(self, X, y=None):
        df = X.copy()

        return df[df.columns[self.selector.get_support(indices=True)]]


class ShiftProcessor(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        self.shift_cols = X.filter(regex="Input_C_[0-9]+_[xy]").columns

        return self

    def transform(self, X, y=None):
        df = X.copy()
        df[self.shift_cols] = np.abs(X[self.shift_cols])

        return df


class A020Processor(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        self.a020_cols = X.filter(regex='Input_A[0-9]_020').columns
        return self

    def transform(self, X, y=None):
        df = X.copy()
        df['A_020_mean'] = X[self.a020_cols].mean(axis=1)
        df['A_020_std'] = X[self.a020_cols].std(axis=1)
        df['A_020_min'] = X[self.a020_cols].min(axis=1)
        df['A_020_max'] = X[self.a020_cols].max(axis=1)

        return df


class DataPreprocessor(Pipeline):
    def __init__(self):
        self.steps = [
            ('variance_selector', VarianceFeatureSelector()),
            ('na_annotation', NAAnnotationTransformer()),
            ('shift_processor', ShiftProcessor()),
            ('fill_na', FillNATransformer()),
            ('quantization', QuantizationTransformer()),
            ('a020_processor', A020Processor()),
            ('outlier_detection', OutlierDetector()),
            ('cluster_maker', ClusterTransformer()),
        ]
        super(DataPreprocessor, self).__init__(steps=self.steps)

    def fit(self, X, y=None, **fit_params):
        super(DataPreprocessor, self).fit(X, y, **fit_params)
