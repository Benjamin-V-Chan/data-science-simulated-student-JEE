import pandas as pd
import os
import matplotlib.pyplot as plt

def load_raw():
    return pd.read_csv(os.path.join('..','data','JEE_Dropout_After_Class_12.csv'))

if __name__ == "__main__":
    fig_dir = os.path.join('..','outputs','figures')
    os.makedirs(fig_dir, exist_ok=True)
    df = load_raw()
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    for col in num_cols:
        plt.figure()
        df[col].hist(bins=30)
        plt.title(f'Distribution of {col}')
        plt.savefig(os.path.join(fig_dir, f'{col}_hist.png'))
        plt.close()
    corr = df[num_cols].corr()
    plt.figure(figsize=(8,6))
    plt.imshow(corr, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(num_cols)), num_cols, rotation=90)
    plt.yticks(range(len(num_cols)), num_cols)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'correlation_matrix.png'))
    plt.close()
