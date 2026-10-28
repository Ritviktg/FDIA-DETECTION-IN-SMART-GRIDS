import numpy as np
import json
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_grid(grid_name, random_state=42):
    X_normal  = np.load(f'dataset/X_normal_{grid_name}.npy')
    X_attacked = np.load(f'dataset/X_attacked_{grid_name}.npy')
    gt_buses   = np.load(f'dataset/gt_buses_{grid_name}.npy')
    
    with open(f'dataset/attack_types_{grid_name}.json') as f:
        attack_types = json.load(f)
    
    n_normal  = len(X_normal)
    n_attacked = len(X_attacked)
    
    X = np.vstack([X_normal, X_attacked])
    y = np.concatenate([
        np.zeros(n_normal),
        np.ones(n_attacked)
    ])
    
    gt_full = np.concatenate([
        np.full(n_normal, -1),
        gt_buses
    ])
    
    at_full = (
        ['normal'] * n_normal + attack_types
    )
    
    indices = np.arange(len(X))
    
    (idx_train, idx_test,
     y_train,   y_test) = train_test_split(
        indices, y,
        test_size=0.2,
        stratify=y,
        random_state=random_state
    )
    
    X_train_raw = X[idx_train]
    X_test_raw  = X[idx_test]
    gt_train    = gt_full[idx_train]
    gt_test     = gt_full[idx_test]
    at_train    = [at_full[i] for i in idx_train]
    at_test     = [at_full[i] for i in idx_test]
    
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)
    
    os.makedirs('dataset', exist_ok=True)
    os.makedirs('models',  exist_ok=True)
    
    np.save(f'dataset/X_train_{grid_name}.npy', X_train)
    np.save(f'dataset/X_test_{grid_name}.npy',  X_test)
    np.save(f'dataset/y_train_{grid_name}.npy', y_train)
    np.save(f'dataset/y_test_{grid_name}.npy',  y_test)
    np.save(f'dataset/gt_train_{grid_name}.npy', gt_train)
    np.save(f'dataset/gt_test_{grid_name}.npy',  gt_test)
    
    with open(f'dataset/at_train_{grid_name}.json','w') as f:
        json.dump(at_train, f)
    with open(f'dataset/at_test_{grid_name}.json','w') as f:
        json.dump(at_test, f)
    
    joblib.dump(scaler, f'models/scaler_{grid_name}.pkl')
    
    print(f"\nGrid {grid_name}-bus preprocessing complete:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_train: {y_train.shape} "
          f"(normal={int((y_train==0).sum())}, "
          f"attack={int((y_train==1).sum())})")
    print(f"  y_test:  {y_test.shape} "
          f"(normal={int((y_test==0).sum())}, "
          f"attack={int((y_test==1).sum())})")
    print(f"  gt_test: {gt_test.shape}")
    
    return X_train, X_test, y_train, y_test, gt_train, gt_test

if __name__ == '__main__':
    data_14  = preprocess_grid('14')
    data_118 = preprocess_grid('118')
    print("\nPreprocessing complete for both grids.")
