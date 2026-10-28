import numpy as np
import shap
import torch
import joblib
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

def run_shap_ft(grid_name, max_explain=100):
    X_train   = np.load(f'dataset/X_train_{grid_name}.npy')
    X_test    = np.load(f'dataset/X_test_{grid_name}.npy')
    y_train   = np.load(f'dataset/y_train_{grid_name}.npy')
    y_pred_ft = np.load(f'dataset/y_pred_ft_{grid_name}.npy')
    
    model = joblib.load(
        f'models/ft_transformer_{grid_name}.pkl'
    )
    model.eval()
    
    X_train_normal = X_train[y_train == 0]
    background     = X_train_normal[:100]
    
    detected_idx = np.where(y_pred_ft == 1)[0]
    
    if len(detected_idx) == 0:
        print(f"No attacks detected for {grid_name}-bus.")
        return None
    
    n_explain  = min(max_explain, len(detected_idx))
    X_explain  = X_test[detected_idx[:n_explain]]
    
    print(f"\nRunning SHAP on {grid_name}-bus...")
    print(f"  Explaining {n_explain} detected samples")
    print(f"  Background: {len(background)} normal samples")
    print(f"  Please wait...")
    
    # Batched predict_fn to prevent massive RAM allocation during SHAP coalition evaluations
    def predict_fn(X):
        batch_size = 256
        probs_list = []
        for i in range(0, len(X), batch_size):
            X_batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32)
            with torch.no_grad():
                logits = model(X_batch, None)
                probs  = torch.softmax(logits, dim=1)
            probs_list.append(probs.numpy())
        return np.concatenate(probs_list, axis=0)
    
    try:
        explainer   = shap.KernelExplainer(
            predict_fn, background
        )
        # Limit coalitions to 100 for dramatic 50x speedup
        shap_values = explainer.shap_values(X_explain, nsamples=100)
        
        # Handle SHAP output structure
        if isinstance(shap_values, list):
            shap_vals_attack = shap_values[1]
        elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
            shap_vals_attack = shap_values[:, :, 1]
        else:
            shap_vals_attack = shap_values
        
        os.makedirs('outputs', exist_ok=True)
        np.save(
            f'outputs/shap_values_{grid_name}.npy',
            shap_vals_attack
        )
        np.save(
            f'outputs/shap_explained_idx_{grid_name}.npy',
            detected_idx[:n_explain]
        )
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_vals_attack, X_explain,
                          plot_type='bar', show=False)
        plt.title(f'SHAP Feature Importance — IEEE {grid_name}-bus')
        plt.tight_layout()
        plt.savefig(f'outputs/shap_bar_{grid_name}bus.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_vals_attack, X_explain,
                          show=False)
        plt.title(f'SHAP Beeswarm — IEEE {grid_name}-bus')
        plt.tight_layout()
        plt.savefig(
            f'outputs/shap_beeswarm_{grid_name}bus.png',
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        
        print(f"  SHAP complete for {grid_name}-bus.")
        print(f"  shap_values shape: {shap_vals_attack.shape}")
        print(f"  Plots saved.")
        
        return shap_vals_attack, detected_idx[:n_explain]
    
    except Exception as e:
        print(f"  SHAP failed for {grid_name}-bus: {e}")
        return None

if __name__ == '__main__':
    shap_result_14  = run_shap_ft('14',  max_explain=100)
    shap_result_118 = run_shap_ft('118', max_explain=30)
