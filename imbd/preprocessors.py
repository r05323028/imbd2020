from sklearn.feature_selection import SelectorMixin
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import Normalizer, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor


class QuantizationTransformer(TransformerMixin):
    '''
    Transform cells into pandas categorical dtype.
    '''
    def __init__(self, unique_count_threshold=5):
        self.unique_count_threshold = unique_count_threshold

    def fit(self, X, y=None, **fit_params):
        uniq = X.nunique()
        mask = uniq[uniq < self.unique_count_threshold]
        self.quant_features = mask.index

        return self

    def transform(self, X):
        df = X.copy()
        df[self.quant_features] = X[self.quant_features].astype('category')

        return df


class NADropper(TransformerMixin, BaseEstimator):
    '''
    Drop NA features.
    '''
    def __init__(self, na_threshold=10):
        self.na_threshold = na_threshold

    def set_params(self, **params):
        super(NADropper, self).set_params(**params)

    def fit(self, X, y=None, **fit_params):
        na_count = X.isnull().sum()
        self.not_na_selector = na_count[na_count < self.na_threshold].index

        return self

    def transform(self, X):
        return X[self.not_na_selector]


class FillNATransformer(TransformerMixin):
    '''
    Filling na cells.
    
    rules:
        category -> mode
        float, int -> mean
    '''
    def fit(self, X, y=None, **fit_params):
        self.float_columns = X.select_dtypes(include=["float"]).columns
        self.category_columns = X.select_dtypes(
            exclude=["int", "float"]).columns
        # self.mode_imputer = SimpleImputer(strategy='most_frequent')
        # self.mean_imputer = SimpleImputer(strategy='mean')

        # knn imputers
        self.float_knn_imputer = KNNImputer()
        self.category_knn_imputer = KNNImputer()
        self.float_knn_imputer.fit(X[self.float_columns])
        self.category_knn_imputer.fit(X[self.category_columns])

        return self

    def transform(self, X):
        df = X.copy()

        # simple imputer
        # df[float_columns] = self.mean_imputer.fit_transform(X[float_columns])
        # df[category_columns] = self.mode_imputer.fit_transform(
        #     X[category_columns])

        # knn transform
        df[self.float_columns] = self.float_knn_imputer.transform(
            X[self.float_columns])
        df[self.category_columns] = self.category_knn_imputer.transform(
            X[self.category_columns])

        return df


class OutlierDetector(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        self.A020_columns = X.filter(regex='Input_A[0-9]+_020').columns

        self.iforest = IsolationForest(n_estimators=1000)
        self.iforest.fit(X[self.A020_columns])

        return self

    def transform(self, X):
        df = X.copy()

        df['outlier'] = self.iforest.predict(X[self.A020_columns])

        return df


class A020Grouper(TransformerMixin):
    def __init__(self, n_groups: int = 2):
        self.n_groups = n_groups

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        df = X.copy()
        # groups = pd.cut(X['A_020_mean'],
        #                 self.n_groups,
        #                 labels=list(range(self.n_groups)))

        groups = [1 if val > 2 else 0 for val in X['A_020_mean'].values]
        groups = tf.one_hot(groups, depth=self.n_groups)
        groups = pd.DataFrame(
            groups.numpy(),
            columns=[f'A_020_group_{i}' for i in range(self.n_groups)])
        groups.index = df.index
        df = pd.concat([df, groups], axis=1)

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

    def transform(self, X):
        df = X.copy()
        clusters = self.model.predict(X)
        clusters = tf.one_hot(clusters, depth=self.n_cluster)
        clusters = pd.DataFrame(
            clusters.numpy(),
            columns=[f'cluster_{i}' for i in range(self.n_cluster)])
        clusters.index = df.index
        df = pd.concat([df, clusters], axis=1)

        return df


class PcaEmbedder(TransformerMixin, BaseEstimator):
    def __init__(self, n_comp: int = 2):
        self.n_comp = n_comp

    def set_params(self, **params):
        super(PcaEmbedder, self).set_params(**params)

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        df = X.copy()
        self.model = PCA(self.n_comp)
        comp = self.model.fit_transform(X)
        comp = pd.DataFrame(comp,
                            columns=[f'comp_{i}' for i in range(self.n_comp)])
        comp.index = df.index
        df = pd.concat([df, comp], axis=1)

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

    def transform(self, X):
        df = X.copy()

        return df[df.columns[self.selector.get_support(indices=True)]]


class NNFeatureEmbedder(TransformerMixin):
    def __init__(self, dropout_rate=0.3):
        self.model = KerasRegressor(build_fn=self.create_model, epochs=30)

    @staticmethod
    def create_model(dropout_rate=0.3):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(20))
        model.compile(loss='mse', optimizer='adam')

        return model

    def fit(self, X, y=None, **fit_params):
        self.model.fit(X, y, verbose=0)

        return self

    def transform(self, X):
        df = X.copy()
        n_cols = pred.shape[1]
        cols = [f'nn_embed_{i}' for i in range(n_cols)]
        df_ext = pd.DataFrame(pred, columns=cols)
        df_ext.index = df.index
        df_ret = pd.concat([df, df_ext], axis=1)

        return df_ret


class ShiftProcessor(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        self.shift_cols = X.filter(regex="Input_C_[0-9]+_[xy]").columns

        return self

    def transform(self, X):
        df = X.copy()
        df[self.shift_cols] = np.abs(X[self.shift_cols])

        return df


class A020Processor(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        self.a020_cols = X.filter(regex='Input_A[0-9]_020').columns
        return self

    def transform(self, X):
        df = X.copy()
        df['A_020_mean'] = X[self.a020_cols].mean(axis=1)
        df['A_020_std'] = X[self.a020_cols].std(axis=1)
        df['A_020_min'] = X[self.a020_cols].min(axis=1)
        df['A_020_max'] = X[self.a020_cols].max(axis=1)

        return df


class ColumnNormalizer(TransformerMixin):
    def __init__(self):
        self.normalizer = Normalizer()

    def fit(self, X, y=None, **fit_params):
        self.normalize_cols = X.filter(regex='Input_A[0-9]_[0-9]+').columns

        return self

    def transform(self, X):
        df = X.copy()
        df[self.normalize_cols] = self.normalizer.transform(
            X[self.normalize_cols])

        return df


class DataPreprocessor(Pipeline):
    def __init__(self):
        self.steps = [
            ('drop_na_by_threshold', NADropper()),
            ('quantization', QuantizationTransformer()),
            ('shift_processor', ShiftProcessor()),
            ('fill_na', FillNATransformer()),
            ('a020_processor', A020Processor()),
            ('variance_selector', VarianceFeatureSelector()),
            ('outlier_detection', OutlierDetector()),
            ('cluster_maker', ClusterTransformer()),
        ]
        super(DataPreprocessor, self).__init__(steps=self.steps)

    def fit(self, X, y=None, **fit_params):
        super(DataPreprocessor, self).fit(X, y, **fit_params)
