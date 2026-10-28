import numpy as np
import pandas as pd
import json

def inspect_grid_dataset(grid_name, count=8):
    print(f"\n================ IEEE {grid_name}-bus Dataset Inspection ================")
    
    # Load files
    X_train = np.load(f'dataset/X_train_{grid_name}.npy')
    X_test  = np.load(f'dataset/X_test_{grid_name}.npy')
    y_train = np.load(f'dataset/y_train_{grid_name}.npy')
    y_test  = np.load(f'dataset/y_test_{grid_name}.npy')
    gt_test = np.load(f'dataset/gt_test_{grid_name}.npy')
    
    with open(f'dataset/at_test_{grid_name}.json') as f:
        at_test = json.load(f)
    
    n_features = X_train.shape[1]
    
    # Grid sizes
    if grid_name == '14':
        n_lines, n_buses = 15, 14
    else:
        n_lines, n_buses = 173, 118
        
    print(f"1. Structure:")
    print(f"  - Total Features:      {n_features} columns")
    print(f"  - Line flow features:  First {n_lines} columns (P_line)")
    print(f"  - Bus power features:   Remaining {n_buses} columns (P_bus)")
    
    print(f"\n2. Data splits:")
    print(f"  - Training Set shape:  {X_train.shape} samples")
    print(f"  - Testing Set shape:   {X_test.shape} samples")
    
    # Build a preview DataFrame of the test dataset (like a CSV view)
    df_rows = []
    # Select columns to display: first 4 features, middle indicator, last feature, and labels
    col_names = [f'Feat_0(P_line)', f'Feat_1', f'Feat_2', '...', f'Feat_{n_features-1}(P_bus)', 'Label(y)', 'Target_Bus(gt)', 'Attack_Profile']
    
    for i in range(min(count, len(X_test))):
        row = [
            round(X_test[i, 0], 4),
            round(X_test[i, 1], 4),
            round(X_test[i, 2], 4),
            '...',
            round(X_test[i, -1], 4),
            int(y_test[i]),
            int(gt_test[i]),
            at_test[i]
        ]
        df_rows.append(row)
        
    preview_df = pd.DataFrame(df_rows, columns=col_names)
    
    print(f"\n3. Spreadsheet Preview of first {count} rows in Test Set:")
    print(preview_df.to_string(index=False))
    print("=" * 90)

if __name__ == '__main__':
    # Force pandas to show all columns for clean console display
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 1000)
    
    inspect_grid_dataset('14', count=10)
    inspect_grid_dataset('118', count=10)
