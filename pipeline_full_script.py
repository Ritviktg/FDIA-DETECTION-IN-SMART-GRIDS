# ==============================================================================
# SHAP-GUIDED ATTACK LOCALIZATION WITH FT-TRANSFORMER FOR EXPLAINABLE FDIA
# DETECTION IN SMART POWER GRIDS (IEEE 14-BUS & IEEE 118-BUS)
# ==============================================================================

# 1. SETUP AND DEPENDENCIES
# Run: pip install -q torch rtdl pandapower shap scikit-learn pyod matplotlib numpy

import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pandapower as pp
import pandapower.networks as pn
import rtdl
import shap

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 2. POWER GRID ENVIRONMENT & PHYSICAL JACOBIAN

def setup_grid_environment(grid_name='14'):
    """
    Initializes grid topology and constructs the physical Jacobian measurement matrix H.
    Includes Base MVA scaling and reactance capping (Contribution 1).
    """
    if grid_name == '14':
        net = pn.case14()
        base_mva = 100.0
    else:
        net = pn.case118()
        base_mva = 1000.0
        
    pp.runpp(net, verbose=False)
    n_buses = len(net.bus)
    n_lines = len(net.line)
    
    # Branch line mapping (from_bus, to_bus)
    line_mapping = {}
    for i, (_, line) in enumerate(net.line.iterrows()):
        f_bus = net.bus.index.get_loc(line.from_bus)
        t_bus = net.bus.index.get_loc(line.to_bus)
        line_mapping[i] = (f_bus, t_bus)
        
    # Construct DC power flow Jacobian H (n_lines + n_buses, n_buses)
    n_meas = n_lines + n_buses
    H = np.zeros((n_meas, n_buses))
    
    # 1. Line power flows P_ij = (theta_i - theta_j) / x_ij
    for i, (_, line) in enumerate(net.line.iterrows()):
        f_bus = net.bus.index.get_loc(line.from_bus)
        t_bus = net.bus.index.get_loc(line.to_bus)
        # Reactance capping at 0.01 ohm to prevent numerical singularities
        x_val = max(float(line.x_ohm), 0.01)
        b_val = (base_mva / x_val)
        H[i, f_bus] = b_val
        H[i, t_bus] = -b_val
        
    # 2. Bus power injections P_i
    for i in range(n_lines):
        f_bus, t_bus = line_mapping[i]
        b_val = H[i, f_bus]
        H[n_lines + f_bus, f_bus] += b_val
        H[n_lines + f_bus, t_bus] -= b_val
        H[n_lines + t_bus, t_bus] += b_val
        H[n_lines + t_bus, f_bus] -= b_val
        
    print(f"[{grid_name}-bus Grid] Buses: {n_buses}, Lines: {n_lines}, Measurement Features: {n_meas}")
    return net, H, line_mapping, n_buses, n_lines, n_meas


# 3. NORMAL STATE DATA GENERATION

def generate_normal_data(net, n_samples=2000, noise_std=0.05):
    """
    Generates normal grid power flow measurements across varying system load conditions.
    """
    n_lines = len(net.line)
    n_buses = len(net.bus)
    n_meas = n_lines + n_buses
    
    base_line_p = net.res_line.p_from_mw.values.copy()
    base_bus_p = net.res_bus.p_mw.values.copy()
    base_z = np.concatenate([base_line_p, base_bus_p])
    
    X_normal = np.zeros((n_samples, n_meas))
    for i in range(n_samples):
        # Realistic load variations with Gaussian noise
        load_variation = 1.0 + np.random.normal(0, noise_std, size=n_meas)
        X_normal[i] = base_z * load_variation
        
    return X_normal


# 4. STEALTHY FDIA ATTACK GENERATION (CONTRIBUTION 1)

def generate_stealthy_fdia(X_normal, H, n_buses, n_lines):
    """
    Generates stealthy False Data Injection Attacks (a = Hc) under 4 attack profiles:
    Pulse, Scaling, Ramping, and Random. Verifies stealth residual constraint.
    """
    n_samples = len(X_normal)
    X_attack = np.zeros_like(X_normal)
    gt_buses = np.zeros(n_samples, dtype=int)
    attack_types = []
    
    profiles = ['pulse', 'scaling', 'ramping', 'random']
    n_per_profile = n_samples // len(profiles)
    
    H_pinv = np.linalg.pinv(H)
    stealth_passed = 0
    
    for i in range(n_samples):
        z_clean = X_normal[i].copy()
        target_bus = np.random.randint(0, n_buses)
        gt_buses[i] = target_bus
        profile = profiles[min(i // n_per_profile, len(profiles) - 1)]
        attack_types.append(profile)
        
        # Define angle perturbation vector c
        c = np.zeros(n_buses)
        if profile == 'pulse':
            c[target_bus] = np.random.choice([-0.25, 0.25])
        elif profile == 'scaling':
            c[target_bus] = 0.20 * np.sign(np.random.randn() or 1.0)
        elif profile == 'ramping':
            step = (i % n_per_profile) / float(n_per_profile)
            c[target_bus] = 0.05 + 0.25 * step
        elif profile == 'random':
            c[target_bus] = np.random.normal(0, 0.15)
            
        # Physical stealth injection a = Hc
        a = H @ c
        z_attacked = z_clean + a
        X_attack[i] = z_attacked
        
        # Verify bad data detection residual: ||r_attacked|| == ||r_clean||
        r_clean = np.linalg.norm(z_clean - H @ (H_pinv @ z_clean))
        r_attack = np.linalg.norm(z_attacked - H @ (H_pinv @ z_attacked))
        if abs(r_attack - r_clean) < 1e-4:
            stealth_passed += 1
            
    print(f"Stealth Verification: {stealth_passed}/{n_samples} passed stealth constraint.")
    return X_attack, gt_buses, attack_types


# 5. DATA PREPROCESSING & DATASET SPLITTING

def preprocess_and_split(X_normal, X_attack, gt_buses, attack_types):
    """
    Constructs balanced dataset (y=0 Normal, y=1 Attack) and applies 80/20 train/test split.
    """
    X = np.vstack([X_normal, X_attack])
    y = np.array([0] * len(X_normal) + [1] * len(X_attack))
    gt = np.concatenate([np.full(len(X_normal), -1), gt_buses])
    all_types = ['normal'] * len(X_normal) + attack_types
    
    indices = np.arange(len(X))
    idx_tr, idx_te = train_test_split(indices, test_size=0.20, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[idx_tr])
    X_test = scaler.transform(X[idx_te])
    
    y_train, y_test = y[idx_tr], y[idx_te]
    gt_train, gt_test = gt[idx_tr], gt[idx_te]
    at_test = [all_types[i] for i in idx_te]
    
    print(f"Training Samples: {len(X_train)} | Testing Samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test, gt_test, at_test, scaler


# 6. FT-TRANSFORMER MODEL ARCHITECTURE (CONTRIBUTION 2)

def build_ft_transformer(n_features, d_token=64, ffn_d_hidden=128):
    """
    Builds the Feature Tokenizer Transformer for tabular grid measurements.
    """
    model = rtdl.FTTransformer.make_baseline(
        n_num_features=n_features,
        cat_cardinalities=[],
        d_token=d_token,
        n_blocks=2,
        attention_dropout=0.2,
        ffn_d_hidden=ffn_d_hidden,
        ffn_dropout=0.1,
        residual_dropout=0.0,
        last_layer_query_idx=[-1],
        d_out=2
    ).to(device)
    return model


# 7. MODEL TRAINING PIPELINE

def train_model(model, X_train, y_train, epochs=25, batch_size=64, lr=3e-4):
    """
    Trains FT-Transformer with AdamW, CrossEntropyLoss, and ReduceLROnPlateau.
    """
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx, None)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)
            
        avg_loss = total_loss / len(X_train)
        scheduler.step(avg_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:2d}/{epochs:2d}] - Loss: {avg_loss:.4f}")
            
    return model


# 8. MODEL EVALUATION & METRICS (FDIA DETECTION)

def evaluate_detection(model, X_test, y_test):
    """
    Evaluates binary detection performance on the test set.
    """
    model.eval()
    with torch.no_grad():
        bx = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = model(bx, None)
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()
        
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1  = f1_score(y_test, y_pred, zero_division=0)
    
    print("\n==============================================")
    print("FDIA DETECTION PERFORMANCE (FT-TRANSFORMER)")
    print("==============================================")
    print(f"Accuracy  (ACC) : {acc:.4f}")
    print(f"Precision (PRE) : {pre:.4f}")
    print(f"Recall    (REC) : {rec:.4f}")
    print(f"F1-Score  (F1)  : {f1:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("==============================================")
    return y_pred


# 9. MULTI-MODEL BENCHMARK COMPARISON

def run_baselines_comparison(X_train, y_train, X_test, y_test, ft_f1):
    """
    Trains standard anomaly detection baselines on normal data and compares F1-scores.
    """
    from pyod.models.lof import LOF
    from pyod.models.ocsvm import OCSVM
    from pyod.models.pca import PCA
    from pyod.models.gmm import GMM
    
    X_train_norm = X_train[y_train == 0]
    baselines = {
        'PCA': PCA(n_components=0.95, random_state=42),
        'LOF': LOF(n_neighbors=20),
        'OCSVM': OCSVM(),
        'GMM': GMM(n_components=2, random_state=42)
    }
    
    results = {'FT-Transformer (Ours)': ft_f1}
    for name, b_model in baselines.items():
        b_model.fit(X_train_norm)
        b_pred = b_model.predict(X_test)
        results[name] = f1_score(y_test, b_pred, zero_division=0)
        
    print("\nBaseline Model F1-Score Comparison:")
    for k, v in results.items():
        print(f"  {k:22s} : {v:.4f}")
    return results


# 10. SHAP KERNEL EXPLAINER ON DETECTED ATTACKS

def compute_shap_explanations(model, X_train, X_test, y_pred, max_explain=50):
    """
    Explains detected attack samples (y_pred == 1) using shap.KernelExplainer.
    """
    detected_attack_idx = np.where(y_pred == 1)[0][:max_explain]
    X_explain = X_test[detected_attack_idx]
    
    # Model prediction wrapper for SHAP
    def predict_proba(x_numpy):
        model.eval()
        with torch.no_grad():
            x_t = torch.tensor(x_numpy, dtype=torch.float32).to(device)
            probs = torch.softmax(model(x_t, None), dim=1)
            return probs[:, 1].cpu().numpy()
            
    # Sample background reference from normal training data
    background = X_train[:50]
    explainer = shap.KernelExplainer(predict_proba, background)
    shap_values = explainer.shap_values(X_explain, nsamples=100, silent=True)
    
    print(f"\nComputed SHAP values for {len(detected_attack_idx)} detected attack instances.")
    return shap_values, detected_attack_idx


# 11. SHAP FEATURE ATTRIBUTION VISUALIZATIONS

def plot_shap_summary(shap_values, X_test, detected_attack_idx, n_lines):
    """
    Generates and saves SHAP summary bar chart of top anomalous grid measurements.
    """
    feature_names = [f"Line_{i}_P" for i in range(n_lines)] + [f"Bus_{j}_P" for j in range(shap_values.shape[1] - n_lines)]
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test[detected_attack_idx], feature_names=feature_names, plot_type="bar", show=False, max_display=10)
    plt.title("Top-10 Anomalous Measurement Attributions (SHAP)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/shap_summary_standalone.png", dpi=150)
    plt.close()
    print("  Saved SHAP feature attribution plot to outputs/shap_summary_standalone.png")


# 12. TOPOLOGICAL ATTACK LOCALIZATION LAYER (CONTRIBUTION 3)

def evaluate_attack_localization(shap_values, detected_attack_idx, gt_test, at_test, line_mapping, n_buses):
    """
    Maps line flow SHAP attributions back to both terminal buses (from_bus, to_bus)
    and evaluates Top-1 and Top-3 attack target localization accuracy.
    """
    top1_matches = []
    top3_matches = []
    
    for i, s_idx in enumerate(detected_attack_idx):
        true_bus = gt_test[s_idx]
        if true_bus == -1:
            continue  # skip false positives
            
        sv = shap_values[i]
        bus_scores = np.zeros(n_buses)
        
        # Map branch SHAP attribution to both connecting terminal nodes
        for line_idx, (f_bus, t_bus) in line_mapping.items():
            if line_idx < len(sv):
                flow_attribution = abs(sv[line_idx])
                bus_scores[f_bus] += flow_attribution
                bus_scores[t_bus] += flow_attribution
                
        ranked_buses = np.argsort(bus_scores)[::-1]
        top1_matches.append(int(true_bus == ranked_buses[0]))
        top3_matches.append(int(true_bus in ranked_buses[:3]))
        
    top1_acc = np.mean(top1_matches) if top1_matches else 0.0
    top3_acc = np.mean(top3_matches) if top3_matches else 0.0
    
    print("\n==============================================")
    print("TOPOLOGICAL ATTACK LOCALIZATION ACCURACY")
    print("==============================================")
    print(f"Top-1 Bus Localization Accuracy : {top1_acc * 100:.2f}%")
    print(f"Top-3 Bus Localization Accuracy : {top3_acc * 100:.2f}%")
    print("==============================================")
    return top1_acc, top3_acc


# 13. SAMPLE LOCALIZATION INFERENCE & RANKING

def display_sample_predictions(shap_values, detected_attack_idx, gt_test, at_test, line_mapping, n_buses, n_display=10):
    """
    Displays tabular sample-by-sample localization predictions with target matching.
    """
    print(f"\n{'Sample':>6} {'Attack Type':>12} {'True Bus':>10} {'Top-1 Pred':>12} {'Top-1 OK?':>10} {'Top-3 Pred':>18} {'Top-3 OK?':>10}")
    print("-" * 84)
    
    count = 0
    for i, s_idx in enumerate(detected_attack_idx):
        if count >= n_display:
            break
        true_bus = gt_test[s_idx]
        atype = at_test[s_idx]
        sv = shap_values[i]
        
        bus_scores = np.zeros(n_buses)
        for line_idx, (f_bus, t_bus) in line_mapping.items():
            if line_idx < len(sv):
                bus_scores[f_bus] += abs(sv[line_idx])
                bus_scores[t_bus] += abs(sv[line_idx])
                
        ranked = np.argsort(bus_scores)[::-1]
        top1 = int(ranked[0])
        top3 = [int(x) for x in ranked[:3]]
        
        if true_bus == -1:
            t1_stat, t3_stat = "❌", "❌"
        else:
            t1_stat = "✅" if true_bus == top1 else "❌"
            t3_stat = "✅" if true_bus in top3 else "❌"
            
        print(f"{count+1:>6} {atype:>12} {true_bus:>10} {top1:>12} {t1_stat:>10} {str(top3):>18} {t3_stat:>10}")
        count += 1


# 14. PHYSICAL ROBUSTNESS & SENSITIVITY TESTS

def evaluate_noise_robustness(model, X_test, y_test, noise_levels=[0.01, 0.05, 0.10, 0.15]):
    """
    Tests FT-Transformer detection robustness against increasing sensor noise.
    """
    print("\nRobustness Evaluation Under Sensor Noise:")
    for noise_std in noise_levels:
        noisy_X = X_test + np.random.normal(0, noise_std, X_test.shape)
        model.eval()
        with torch.no_grad():
            bx = torch.tensor(noisy_X, dtype=torch.float32).to(device)
            preds = torch.argmax(model(bx, None), dim=1).cpu().numpy()
        f1 = f1_score(y_test, preds, zero_division=0)
        print(f"  Noise σ = {noise_std:4.2f} | F1-Score = {f1:.4f}")


# 15. COMPUTATIONAL PERFORMANCE & SCADA LATENCY BENCHMARK

def benchmark_inference_latency(model, X_test, n_trials=500):
    """
    Measures inference latency and throughput for real-time SCADA RTU/PMU cycles.
    """
    sample = torch.tensor(X_test[:1], dtype=torch.float32).to(device)
    model.eval()
    
    # GPU warm-up
    with torch.no_grad():
        for _ in range(50):
            _ = model(sample, None)
            
    latencies = []
    with torch.no_grad():
        for _ in range(n_trials):
            t_start = time.perf_counter()
            _ = model(sample, None)
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000)  # to milliseconds
            
    avg_lat = np.mean(latencies)
    med_lat = np.median(latencies)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)
    fps = 1000.0 / avg_lat
    
    print("\n==============================================")
    print("COMPUTATIONAL PERFORMANCE & LATENCY")
    print("==============================================")
    print(f"Average Latency : {avg_lat:.3f} ms")
    print(f"Median Latency  : {med_lat:.3f} ms")
    print(f"Minimum Latency : {min_lat:.3f} ms")
    print(f"Maximum Latency : {max_lat:.3f} ms")
    print(f"Throughput      : {fps:.1f} samples/sec")
    print(f"SCADA Ready (<4s cycle): {'YES ✅' if avg_lat < 4000 else 'NO ❌'}")
    print("==============================================")
    print("\nPipeline execution successfully completed.")


# ==============================================================================
# MAIN EXECUTION ROUTINE
# ==============================================================================

if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    
    # Steps 2-5: Grid environment, data generation, and preprocessing
    net, H, line_mapping, n_buses, n_lines, n_meas = setup_grid_environment('14')
    X_norm = generate_normal_data(net, n_samples=1000)
    X_att, gt_buses, attack_types = generate_stealthy_fdia(X_norm, H, n_buses, n_lines)
    X_tr, X_te, y_tr, y_te, gt_te, at_te, scaler = preprocess_and_split(X_norm, X_att, gt_buses, attack_types)
    
    # Steps 6-8: Model training and detection
    model = build_ft_transformer(n_features=n_meas, d_token=64, ffn_d_hidden=128)
    model = train_model(model, X_tr, y_tr, epochs=15, batch_size=64)
    y_pred = evaluate_detection(model, X_te, y_te)
    
    # Step 9: Multi-model comparison
    _ = run_baselines_comparison(X_tr, y_tr, X_te, y_te, ft_f1=f1_score(y_te, y_pred, zero_division=0))
    
    # Steps 10-11: SHAP analysis & plotting
    shap_vals, det_idx = compute_shap_explanations(model, X_tr, X_te, y_pred, max_explain=30)
    plot_shap_summary(shap_vals, X_te, det_idx, n_lines)
    
    # Steps 12-13: Attack localization layer & sample predictions
    evaluate_attack_localization(shap_vals, det_idx, gt_te, at_te, line_mapping, n_buses)
    display_sample_predictions(shap_vals, det_idx, gt_te, at_te, line_mapping, n_buses, n_display=10)
    
    # Steps 14-15: Robustness and SCADA latency performance
    evaluate_noise_robustness(model, X_te, y_te)
    benchmark_inference_latency(model, X_te)
