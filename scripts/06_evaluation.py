# scripts/06_evaluation.py
# Description: Load best models + test set; plot ROC curves + confusion matrices
# import joblib, os, matplotlib.pyplot, sklearn.metrics
#
# function load_all():
#     load X_test,y_test and both best models
#     return models, X_test, y_test
#
# if __name__ == "__main__":
#     models,X_test,y_test = load_all()
#     for each model:
#         compute fpr,tpr; plot ROC → save
#         compute confusion matrix; plot → save
