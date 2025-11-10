import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

def basic_preprocess(df):
    # target
    y = df['SalePrice'] if 'SalePrice' in df else None
    X = df.copy()
    if 'SalePrice' in X: X = X.drop(columns=['SalePrice'])
    # select a subset of features for simplicity:
    keep = ['OverallQual','GrLivArea','YearBuilt','TotalBsmtSF','FullBath','GarageCars','LotArea','Neighborhood']
    X = X[keep]
    # Impute numeric with median
    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                         ('ohe', OneHotEncoder(handle_unknown='ignore'))])
    preproc = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])
    X_proc = preproc.fit_transform(X)
    return X_proc, y, preproc
