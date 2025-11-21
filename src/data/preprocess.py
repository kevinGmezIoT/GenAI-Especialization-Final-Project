import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

def create_preprocessor():
    """
    Creates a Scikit-Learn ColumnTransformer to preprocess the data.
    """
    categorical_cols = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
    numerical_cols = ["Age", "Job", "Credit amount", "Duration"]
    
    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    
    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        verbose_feature_names_out=False
    )
    
    return preprocessor

def load_data(path):
    """
    Loads data from a CSV file.
    """
    return pd.read_csv(path)
