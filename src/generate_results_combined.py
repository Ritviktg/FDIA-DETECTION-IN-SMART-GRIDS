import pandas as pd
import numpy as np
import os

def generate_combined_table():
    print("Generating combined results table...")
    
    # Load detection results
    det_14  = pd.read_csv('results/detection_14bus.csv')
    det_118 = pd.read_csv('results/detection_118bus.csv')
    
    # Load localization results
    loc_14  = pd.read_csv('results/localization_14bus.csv')
    loc_118 = pd.read_csv('results/localization_118bus.csv')
    
    # Extract overall localization accuracies for Ours
    overall_loc_14 = loc_14[loc_14['Attack_Type'] == 'Overall'].iloc[0]
    overall_loc_118 = loc_118[loc_118['Attack_Type'] == 'Overall'].iloc[0]
    
    top1_14_val = round(overall_loc_14['Top1_Accuracy'], 4)
    top3_14_val = round(overall_loc_14['Top3_Accuracy'], 4)
    top1_118_val = round(overall_loc_118['Top1_Accuracy'], 4)
    top3_118_val = round(overall_loc_118['Top3_Accuracy'], 4)
    
    # Normalize LUNAR model name for clean row merging
    det_14['Model'] = det_14['Model'].replace('LUNAR (k=5)', 'LUNAR')
    det_118['Model'] = det_118['Model'].replace('LUNAR (k=130)', 'LUNAR')

    # Rename columns for merging
    det_14.rename(columns={
        'ACC': 'ACC_14',
        'PRE': 'PRE_14',
        'REC': 'REC_14',
        'F1':  'F1_14'
    }, inplace=True)
    
    det_118.rename(columns={
        'ACC': 'ACC_118',
        'PRE': 'PRE_118',
        'REC': 'REC_118',
        'F1':  'F1_118'
    }, inplace=True)
    
    # Merge on Model column
    combined = pd.merge(det_14, det_118, on='Model', how='outer')
    
    # Add localization columns initialized to 'N/A'
    combined['Top1_14'] = 'N/A'
    combined['Top3_14'] = 'N/A'
    combined['Top1_118'] = 'N/A'
    combined['Top3_118'] = 'N/A'
    
    # Assign localization values to Ours
    ours_mask = combined['Model'].str.contains('FT-Transformer')
    combined.loc[ours_mask, 'Top1_14'] = str(top1_14_val)
    combined.loc[ours_mask, 'Top3_14'] = str(top3_14_val)
    combined.loc[ours_mask, 'Top1_118'] = str(top1_118_val)
    combined.loc[ours_mask, 'Top3_118'] = str(top3_118_val)
    
    # Format detection columns as float and round to 4 decimals
    det_cols = ['ACC_14', 'PRE_14', 'REC_14', 'F1_14', 'ACC_118', 'PRE_118', 'REC_118', 'F1_118']
    for col in det_cols:
        combined[col] = combined[col].astype(float).round(4)
        
    combined.to_csv('results/results_combined.csv', index=False)
    
    print("\n===== Final Master Results Table =====")
    print(combined.to_string(index=False))
    print("\nSaved combined table to results/results_combined.csv.")

if __name__ == '__main__':
    generate_combined_table()
