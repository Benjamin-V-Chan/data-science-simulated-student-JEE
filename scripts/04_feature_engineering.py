import os
import joblib
from sklearn.preprocessing import PolynomialFeatures

def load_splits():
    return joblib.load(os.path.join('..','outputs','data_splits.pkl'))

def expand(X_train, X_test):
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_fe = poly.fit_transform(X_train)
    X_test_fe  = poly.transform(X_test)
    return X_train_fe, X_test_fe, poly

if __name__ == "__main__":
    os.makedirs(os.path.join('..','outputs'), exist_ok=True)
    X_train, X_test, y_train, y_test = load_splits()
    X_train_fe, X_test_fe, poly = expand(X_train, X_test)
    joblib.dump(poly, os.path.join('..','outputs','feature_engineer.pkl'))
    joblib.dump((X_train_fe, X_test_fe, y_train, y_test),
                os.path.join('..','outputs','data_fe_splits.pkl'))
