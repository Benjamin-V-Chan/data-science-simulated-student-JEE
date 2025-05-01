# scripts/05_model_training.py
# Description: Grid-search LogisticRegression + RandomForest; handle imbalance; save best models + perf
# import joblib, os, pandas
# from sklearn...
#
# function load_fe_splits():
#     return joblib.load('outputs/data_fe_splits.pkl')
#
# if __name__ == "__main__":
#     X_train,X_test,y_train,y_test = load_fe_splits()
#     define models dict and param grids
#     for each model:
#         run GridSearchCV(cv=5,scoring='roc_auc')
#         save grid + best_estimator to outputs/models
#     compute test ROC-AUC for each → save outputs/model_performance.csv
