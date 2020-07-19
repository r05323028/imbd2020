from sklearn.feature_selection import SelectorMixin
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer


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
        self.mode_imputer = SimpleImputer(strategy='most_frequent')
        self.mean_imputer = SimpleImputer(strategy='mean')
        self.knn_imputer = KNNImputer()

        return self

    def transform(self, X):
        df = X.copy()
        float_columns = df.select_dtypes(include=["float"]).columns
        category_columns = df.select_dtypes(exclude=["int", "float"]).columns

        # simple imputer
        # df[float_columns] = self.mean_imputer.fit_transform(X[float_columns])
        # df[category_columns] = self.mode_imputer.fit_transform(
        #     X[category_columns])

        # knn imputer
        df[float_columns] = self.knn_imputer.fit_transform(X[float_columns])
        df[category_columns] = self.knn_imputer.fit_transform(
            X[category_columns])

        return df