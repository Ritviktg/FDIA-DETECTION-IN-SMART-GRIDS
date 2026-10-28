import numpy as np
import pandas as pd
import joblib
import os
import warnings
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score)
from pyod.models.lunar import LUNAR
from pyod.models.lof   import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.pca   import PCA
from pyod.models.gmm   import GMM
from pyod.models.cof   import COF

warnings.filterwarnings('ignore')

def evaluate_model(name, y_test, y_pred):
    return {
        'Model': name,
        'ACC': round(accuracy_score(y_test, y_pred),  4),
        'PRE': round(precision_score(y_test, y_pred,
                                     zero_division=0), 4),
        'REC': round(recall_score(y_test, y_pred,
                                  zero_division=0), 4),
        'F1':  round(f1_score(y_test, y_pred,
                              zero_division=0), 4)
    }

def train_and_compare(grid_name, ft_results):
    X_train = np.load(f'dataset/X_train_{grid_name}.npy')
    X_test  = np.load(f'dataset/X_test_{grid_name}.npy')
    y_train = np.load(f'dataset/y_train_{grid_name}.npy')
    y_test  = np.load(f'dataset/y_test_{grid_name}.npy')
    
    X_train_normal = X_train[y_train == 0]
    
    k_neighbours = 5 if grid_name == '14' else 130
    
    models = {
        'LOF':   LOF(),
        'OCSVM': OCSVM(),
        'PCA':   PCA(n_components=0.95, random_state=42),
        'GMM':   GMM(),
        'COF':   COF(),
        f'LUNAR (k={k_neighbours})': LUNAR(
            n_neighbours=k_neighbours
        )
    }
    
    rows = []
    for name, model in models.items():
        try:
            print(f"  Training {name}...")
            model.fit(X_train_normal)
            y_pred = model.predict(X_test)
            rows.append(evaluate_model(name, y_test, y_pred))
            joblib.dump(model,
                f'models/{name.replace(" ","_").replace("=","")}'
                f'_{grid_name}.pkl')
        except Exception as e:
            print(f"  {name} failed: {e}")
    
    ft_row = {
        'Model': 'FT-Transformer + SHAP (Ours)',
        'ACC': ft_results['ACC'],
        'PRE': ft_results['PRE'],
        'REC': ft_results['REC'],
        'F1':  ft_results['F1']
    }
    rows.append(ft_row)
    
    df = pd.DataFrame(rows)
    
    os.makedirs('results', exist_ok=True)
    df.to_csv(f'results/detection_{grid_name}bus.csv',
              index=False)
    
    print(f"\n===== Detection Results — {grid_name}-bus =====")
    print(df.to_string(index=False))
    
    return df

if __name__ == '__main__':
    # Exact FT-Transformer results as specified by the user
    ft_results_14 = {
        'ACC': 0.9151,
        'PRE': 1.0,
        'REC': 0.8303,
        'F1':  0.9073
    }
    ft_results_118 = {
        'ACC': 0.8732,
        'PRE': 0.8880,
        'REC': 0.8542,
        'F1':  0.8708
    }

    print("Training baselines for 14-bus...")
    df_14  = train_and_compare('14',  ft_results_14)

    print("\nTraining baselines for 118-bus...")
    df_118 = train_and_compare('118', ft_results_118)

    print("\nBaseline comparison complete.")
