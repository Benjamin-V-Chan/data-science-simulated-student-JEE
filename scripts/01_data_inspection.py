# scripts/01_data_inspection.py
# Description: Load raw dataset and produce summary statistics + missing value counts
# import pandas, os
#
# function load_data():
#     path = os.path.join('data','JEE_Dropout_After_Class_12.csv')
#     df = pd.read_csv(path)
#     return df
#
# function summarize_data(df):
#     stats = df.describe(include='all').T
#     stats['missing'] = df.isnull().sum()
#     return stats
#
# if __name__ == "__main__":
#     df = load_data()
#     summary = summarize_data(df)
#     summary.to_csv('outputs/data_summary.csv')
