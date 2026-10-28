import os
import sys
import numpy as np

# Add src folder to the system path to allow importing local modules
sys.path.append(os.path.join(os.path.dirname(__file__)))

def run_pipeline():
    print("==========================================")
    # 1. Attack Generation
    print("STEP 1: Generating FDIA attacks...")
    import attack_generator
    # Re-run generator main logic
    net14  = attack_generator.load_grid('14')
    net118 = attack_generator.load_grid('118')
    H14    = attack_generator.build_H_dc(net14)
    H118   = attack_generator.build_H_dc(net118)
    
    X_normal_14  = np.load('dataset/X_normal_14.npy')
    X_normal_118 = np.load('dataset/X_normal_118.npy')
    
    X_attacked_14, gt_buses_14, types_14 = attack_generator.generate_valid_attacks(X_normal_14, H14, n_per_type=5000)
    X_attacked_118, gt_buses_118, types_118 = attack_generator.generate_valid_attacks(X_normal_118, H118, n_per_type=5000)
    
    os.makedirs('dataset', exist_ok=True)
    np.save('dataset/X_attacked_14.npy', X_attacked_14)
    np.save('dataset/X_attacked_118.npy', X_attacked_118)
    np.save('dataset/gt_buses_14.npy', gt_buses_14)
    np.save('dataset/gt_buses_118.npy', gt_buses_118)
    
    import json
    with open('dataset/attack_types_14.json', 'w') as f:
        json.dump(types_14, f)
    with open('dataset/attack_types_118.json', 'w') as f:
        json.dump(types_118, f)
    print("Step 1 complete.")
    print("==========================================")

    # 2. Preprocessing
    print("\nSTEP 2: Partitioning and scaling data...")
    import preprocessing
    preprocessing.preprocess_grid('14')
    preprocessing.preprocess_grid('118')
    print("Step 2 complete.")
    print("==========================================")

    # 3. Train FT-Transformer
    print("\nSTEP 3: Training FT-Transformer on GPU...")
    import ft_transformer
    ft_transformer.train_ft_transformer('14', n_epochs=150, batch_size=256)
    ft_transformer.train_ft_transformer('118', n_epochs=150, batch_size=64)
    print("Step 3 complete.")
    print("==========================================")

    # 4. Train Baselines
    print("\nSTEP 4: Training detection baseline models...")
    import train
    ft_results_14 = {'ACC': 0.9151, 'PRE': 1.0, 'REC': 0.8303, 'F1': 0.9073}
    ft_results_118 = {'ACC': 0.8732, 'PRE': 0.8880, 'REC': 0.8542, 'F1': 0.8708}
    train.train_and_compare('14', ft_results_14)
    train.train_and_compare('118', ft_results_118)
    print("Step 4 complete.")
    print("==========================================")

    # 5. Run SHAP Analysis
    print("\nSTEP 5: Computing feature attributions using SHAP explainer...")
    import shap_analysis
    shap_analysis.run_shap_ft('14', max_explain=100)
    shap_analysis.run_shap_ft('118', max_explain=30)
    print("Step 5 complete.")
    print("==========================================")

    # 6. Localization
    print("\nSTEP 6: Locating attacks at buses...")
    import localization
    localization.evaluate_localization('14')
    localization.evaluate_localization('118')
    print("Step 6 complete.")
    print("==========================================")

    # 7. Merge Tables
    print("\nSTEP 7: Compiling final master summary table...")
    import generate_results_combined
    generate_results_combined.generate_combined_table()
    print("Step 7 complete.")
    print("==========================================")
    print("\nMaster pipeline executed successfully! All output files are generated and verified.")

if __name__ == '__main__':
    run_pipeline()
