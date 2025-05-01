import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

def load_artifacts():
    pre = joblib.load(os.path.join('..','outputs','preprocessor.pkl'))
    rf  = joblib.load(os.path.join('..','outputs','models','random_forest_best_model.pkl'))
    return pre, rf

if __name__ == "__main__":
    fig_dir = os.path.join('..','outputs','figures')
    os.makedirs(fig_dir, exist_ok=True)
    pre, rf = load_artifacts()
    feat_names = pre.get_feature_names_out()
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1][:20]
    top_feats = feat_names[idx]
    top_imp   = importances[idx]
    plt.figure(figsize=(6,8))
    plt.barh(range(len(top_imp)), top_imp[::-1])
    plt.yticks(range(len(top_feats)), top_feats[::-1])
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'feature_importance.png'))
    plt.close()
