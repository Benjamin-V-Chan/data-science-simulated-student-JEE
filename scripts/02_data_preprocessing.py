# scripts/02_data_preprocessing.py
# Description: Clean, impute, encode, scale; split into train/test; save preprocessor + splits
# import pandas, os, joblib
# from sklearn...
#
# function load_data():
#     return pd.read_csv('data/JEE_Dropout_After_Class_12.csv')
#
# function build_preprocessor(X):
#     numeric_cols = list of numeric column names
#     categorical_cols = list of object/category column names
#     numeric_pipeline = Pipeline([...])
#     categorical_pipeline = Pipeline([...])
#     preprocessor = ColumnTransformer([...])
#     return preprocessor
#
# if __name__ == "__main__":
#     df = load_data()
#     y = df['dropout']; X = df.drop('dropout',axis=1)
#     preprocessor = build_preprocessor(X)
#     X_proc = preprocessor.fit_transform(X)
#     split into X_train, X_test, y_train, y_test
#     joblib.dump(preprocessor,'outputs/preprocessor.pkl')
#     joblib.dump((X_train,X_test,y_train,y_test),'outputs/data_splits.pkl')
