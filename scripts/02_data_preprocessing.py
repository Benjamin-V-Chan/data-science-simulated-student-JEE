import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data():
    return pd.read_csv(os.path.join('..','data','JEE_Dropout_After_Class_12.csv'))

def build_preprocessor(X):
    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
    num_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])
    return preprocessor

if __name__ == "__main__":
    os.makedirs(os.path.join('..','outputs'), exist_ok=True)
    df = load_data()
    y = df['dropout']
    X = df.drop('dropout', axis=1)
    pre = build_preprocessor(X)
    X_proc = pre.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.2, random_state=42, stratify=y
    )
    joblib.dump(pre, os.path.join('..','outputs','preprocessor.pkl'))
    joblib.dump((X_train, X_test, y_train, y_test),
                os.path.join('..','outputs','data_splits.pkl'))
