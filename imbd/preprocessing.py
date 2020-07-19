from sklearn.feature_selection import SelectorMixin
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class QuantizationTransformer(TransformerMixin):
    '''
    Transform cells into pandas categorical dtype.
    '''
    unique_count_threshold = 5

    def fit(self, X, y=None):
        uniq = X.nunique()
        mask = uniq[uniq < self.unique_count_threshold]
        self.quant_features = mask.index

        return self

    def transform(self, X):
        df = X.copy()
        df[self.quant_features] = X[self.quant_features].astype('category')

        return df


class FeaturesSelector(SelectorMixin):
    '''
    Drop NA & N-unique = 1 features (by column).
    '''
    na_count_threshold = 10
    unique_count_threshold = 2

    def inverse_transform(self):
        return self

    def _get_support_mask(self):
        return NotImplemented

    def fit(self, X, y=None):
        na_count = X.isnull().sum()
        mask = na_count[na_count < self.na_count_threshold].index
        not_na_selector = mask
        uniq = X.nunique()
        not_uniq_selector = uniq[uniq > self.unique_count_threshold].index

        self.features_selector = not_na_selector & not_uniq_selector

        return self

    def transform(self, X):
        return X[self.features_selector]


class FillNATransformer(TransformerMixin):
    '''
    Filling na cells.
    
    rules:
        category -> mode
        float, int -> mean
    '''
    def fit(self, X):
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
    def fit(self, X):
        self.A_columns = X.filter(
            regex='(Input_A[0-9]+_[0-9]+|Output_A[0-9]+)').columns

        # self.iforest = IsolationForest(n_estimators=1000)
        # # self.iforest.fit(X)
        # self.iforest.fit(X[self.A_columns])

        # local outlier factor
        self.lof = LocalOutlierFactor(n_neighbors=10, novelty=True)
        self.lof.fit(X)

        return self

    def transform(self, X):
        df = X.copy()
        # df['outlier'] = self.iforest.predict(X[self.A_columns])
        # df['outlier'] = self.iforest.predict(X)

        df['outlier'] = self.lof.predict(X)

        return df


class VarianceFeatureSelector(TransformerMixin):
    threshold = 0.2

    def fit(self, X):
        self.selector = VarianceThreshold(self.threshold)
        self.selector.fit(X)

        return self

    def transform(self, X):
        df = X.copy()

        return df[df.columns[self.selector.get_support(indices=True)]]