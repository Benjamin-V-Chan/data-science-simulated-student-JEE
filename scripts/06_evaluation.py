import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

def load_all():
    splits = joblib.load(os.path.join('..','outputs','data_fe_splits.pkl'))
    models = {
        'logistic': joblib.load(os.path.join('..','outputs','models','logistic_best_model.pkl')),
        'random_forest': joblib.load(os.path.join('..','outputs','models','random_forest_best_model.pkl'))
    }
    return models, splits[1], splits[3]  # X_test, y_test

if __name__ == "__main__":
    fig_dir = os.path.join('..','outputs','figures')
    os.makedirs(fig_dir, exist_ok=True)
    models, X_test, y_test = load_all()
    for name, model in models.items():
        y_score = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0,1],[0,1],'--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend()
        plt.savefig(os.path.join(fig_dir, f'{name}_roc.png'))
        plt.close()

        cm = confusion_matrix(y_test, model.predict(X_test))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f'Confusion Matrix - {name}')
        plt.savefig(os.path.join(fig_dir, f'{name}_cm.png'))
        plt.close()
