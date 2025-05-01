# scripts/04_feature_engineering.py
# Description: Expand numeric features with pairwise interactions (degree=2)
# import joblib, os
# from sklearn.preprocessing import PolynomialFeatures
#
# function load_splits():
#     return joblib.load('outputs/data_splits.pkl')
#
# function expand(X_train,X_test):
#     poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
#     X_train_fe = poly.fit_transform(X_train)
#     X_test_fe = poly.transform(X_test)
#     return X_train_fe, X_test_fe, poly
#
# if __name__ == "__main__":
#     X_train,X_test,y_train,y_test = load_splits()
#     X_train_fe,X_test_fe,poly = expand(X_train,X_test)
#     joblib.dump(poly,'outputs/feature_engineer.pkl')
#     joblib.dump((X_train_fe,X_test_fe,y_train,y_test),'outputs/data_fe_splits.pkl')
