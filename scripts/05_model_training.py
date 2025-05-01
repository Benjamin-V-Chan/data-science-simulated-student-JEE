import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

def load_fe_splits():
    return joblib.load(os.path.join('..','outputs','data_fe_splits.pkl'))

if __name__ == "__main__":
    os.makedirs(os.path.join('..','outputs','models'), exist_ok=True)
    X_train, X_test, y_train, y_test = load_fe_splits()
    models = {
        'logistic': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'random_forest': RandomForestClassifier(class_weight='balanced')
    }
    params = {
        'logistic': {'C': [0.01, 0.1, 1, 10]},
        'random_forest': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
    }
    best = {}
    for name in models:
        grid = GridSearchCV(models[name], params[name], cv=5, scoring='roc_auc')
        grid.fit(X_train, y_train)
        best[name] = grid.best_estimator_
        joblib.dump(grid,     os.path.join('..','outputs','models',f'{name}_grid.pkl'))
        joblib.dump(grid.best_estimator_,
                    os.path.join('..','outputs','models',f'{name}_best_model.pkl'))
    results = {n: roc_auc_score(y_test, best[n].predict_proba(X_test)[:,1])
               for n in best}
    pd.Series(results).to_csv(os.path.join('..','outputs','model_performance.csv'))
