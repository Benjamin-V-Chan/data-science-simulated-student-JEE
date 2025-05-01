import pandas as pd
import os

def load_data():
    path = os.path.join('..','data','JEE_Dropout_After_Class_12.csv')
    return pd.read_csv(path)

def summarize_data(df):
    summary = df.describe(include='all').T
    summary['missing'] = df.isnull().sum()
    return summary

if __name__ == "__main__":
    os.makedirs(os.path.join('..','outputs'), exist_ok=True)
    df = load_data()
    summary = summarize_data(df)
    summary.to_csv(os.path.join('..','outputs','data_summary.csv'))
