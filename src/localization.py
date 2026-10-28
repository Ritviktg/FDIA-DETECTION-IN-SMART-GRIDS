import numpy as np
import pandas as pd
import os
import json
import pandapower as pp
import pandapower.networks as pn

def load_grid_line_mapping(grid_name):
    if grid_name == '14':
        net = pn.case14()
    else:  # 118
        net = pn.case118()
    pp.runpp(net, verbose=False)
    
    # Map each line index to its (from_bus_idx, to_bus_idx)
    line_mapping = {}
    for i, (_, line) in enumerate(net.line.iterrows()):
        try:
            from_idx = net.bus.index.get_loc(line.from_bus)
            to_idx   = net.bus.index.get_loc(line.to_bus)
            line_mapping[i] = (from_idx, to_idx)
        except Exception as e:
            pass
    return line_mapping, len(net.bus)

def evaluate_localization(grid_name):
    print(f"\nRunning Attack Localization for IEEE {grid_name}-bus...")
    
    # Load SHAP values and explained indices
    shap_vals = np.load(f'outputs/shap_values_{grid_name}.npy')
    explained_idx = np.load(f'outputs/shap_explained_idx_{grid_name}.npy')
    
    # Load ground truth buses and attack types
    gt_test = np.load(f'dataset/gt_test_{grid_name}.npy')
    true_buses = gt_test[explained_idx]
    
    with open(f'dataset/at_test_{grid_name}.json') as f:
        at_test = json.load(f)
    attack_types = [at_test[idx] for idx in explained_idx]
    
    # Filter out false positives (normal samples that were predicted as attacks)
    # because normal samples do not have a physical target bus label (gt = -1) to localize.
    valid_mask = (true_buses != -1)
    shap_vals = shap_vals[valid_mask]
    true_buses = true_buses[valid_mask]
    attack_types = [attack_types[j] for j in range(len(valid_mask)) if valid_mask[j]]
    
    line_mapping, n_buses = load_grid_line_mapping(grid_name)
    n_lines = len(line_mapping)
    
    top1_matches = []
    top3_matches = []
    
    for i in range(len(shap_vals)):
        s = shap_vals[i]
        true_bus = true_buses[i]
        
        # Calculate bus SHAP attribution by summing absolute line SHAP values connected to each bus
        bus_attributions = np.zeros(n_buses)
        for line_idx, (from_bus, to_bus) in line_mapping.items():
            if line_idx < len(s):
                val = abs(s[line_idx])
                bus_attributions[from_bus] += val
                bus_attributions[to_bus]   += val
        
        # Sort buses in descending order of attribution
        sorted_buses = np.argsort(bus_attributions)[::-1]
        
        top1_matches.append(int(true_bus == sorted_buses[0]))
        top3_matches.append(int(true_bus in sorted_buses[:3]))
        
    top1_acc = np.mean(top1_matches)
    top3_acc = np.mean(top3_matches)
    
    print(f"  Top-1 Accuracy: {top1_acc:.4f}")
    print(f"  Top-3 Accuracy: {top3_acc:.4f}")
    
    # Group results by attack type
    df_samples = pd.DataFrame({
        'Attack_Type': attack_types,
        'Top1_Match': top1_matches,
        'Top3_Match': top3_matches
    })
    
    summary = df_samples.groupby('Attack_Type').mean().reset_index()
    summary.rename(columns={
        'Top1_Match': 'Top1_Accuracy',
        'Top3_Match': 'Top3_Accuracy'
    }, inplace=True)
    
    # Add Overall row
    overall_row = pd.DataFrame([{
        'Attack_Type': 'Overall',
        'Top1_Accuracy': top1_acc,
        'Top3_Accuracy': top3_acc
    }])
    summary = pd.concat([summary, overall_row], ignore_index=True)
    
    os.makedirs('results', exist_ok=True)
    summary.to_csv(f'results/localization_{grid_name}bus.csv', index=False)
    
    print(f"\n===== Localization Results — {grid_name}-bus =====")
    print(summary.to_string(index=False))
    
    return summary

if __name__ == '__main__':
    evaluate_localization('14')
    evaluate_localization('118')
