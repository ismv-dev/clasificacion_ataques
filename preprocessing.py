import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class Limpiador(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.str_cols_ = []
        self.num_cols_ = []
        self.cols_to_drop_ = []
        self.encoded_feature_names_ = []

    def _prepare_df(self, X):
        X = X.copy()
        X.columns = X.columns.astype(str)
        
        str_type_cols = X.select_dtypes(include=['object', 'string']).columns
        if not str_type_cols.empty:
            X[str_type_cols] = X[str_type_cols].apply(lambda x: x.str.strip(" '"))
        return X

    def fit(self, X, y=None):
        X = self._prepare_df(X)
        
        self.cols_to_drop_ = [col for col in X.columns if X[col].nunique() <= 1]
        remaining_cols = [c for c in X.columns if c not in self.cols_to_drop_]
        
        self.str_cols_ = []
        self.num_cols_ = []
        
        for col in remaining_cols:
            converted = pd.to_numeric(X[col], errors='coerce')
            if converted.isna().all():
                self.str_cols_.append(col)
            else:
                self.num_cols_.append(col)
        
        if self.str_cols_:
            self.encoder.fit(X[self.str_cols_].astype(str))
            self.encoded_feature_names_ = self.encoder.get_feature_names_out(self.str_cols_)
            
        return self

    def transform(self, X):
        X = self._prepare_df(X)
        X_num = X[self.num_cols_].apply(pd.to_numeric, errors='coerce')
        if self.str_cols_:
            encoded_data = self.encoder.transform(X[self.str_cols_].astype(str))
            X_encoded_df = pd.DataFrame(
                encoded_data, 
                columns=self.encoded_feature_names_, 
                index=X.index
            )
            return pd.concat([X_num, X_encoded_df], axis=1)
        return X_num