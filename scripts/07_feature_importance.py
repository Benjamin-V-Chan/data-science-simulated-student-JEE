# scripts/07_feature_importance.py
# Description: Extract feature‐names from preprocessor; plot top 20 importances from RF
# import joblib, os, numpy, matplotlib.pyplot
#
# function load_artifacts():
#     pre = joblib.load('outputs/preprocessor.pkl')
#     rf = joblib.load('outputs/models/random_forest_best_model.pkl')
#     return pre, rf
#
# if __name__ == "__main__":
#     pre, rf = load_artifacts()
#     names = pre.get_feature_names_out()
#     importances = rf.feature_importances_
#     select top 20; barh plot → save
