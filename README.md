# SHAP-Guided Attack Localization with FT-Transformer for Explainable FDIA Detection in Smart Grids

This repository implements a three-part explainable FDIA framework:
1. **Valid Stealthy Attack Generation**: Constructing $a = Hc$ target injections on line flow measurements with dynamic grid capacity scaling and reactance capping.
2. **Tabular Deep Learning**: GPU-accelerated FT-Transformer with plateaud learning scheduling and early stopping to outperform baseline detectors.
3. **SHAP-Guided Localization**: A physical bus localization mapping layer utilizing absolute SHAP flows to pinpoint the source of injected attacks.

---

## Installation

Ensure your Python environment contains CUDA-enabled PyTorch (for GPU acceleration of the FT-Transformer) and all dependency packages:

```bash
pip install -r requirements.txt
```

---

## How to Run the Project

### Option A: Run the entire pipeline with a single command (Recommended)
You can run the entire pipeline from end-to-end sequentially using the master execution script:
```bash
python src/main.py
```

### Option B: Run step-by-step
Alternatively, the pipeline is modularized and can be executed step-by-step in the following sequence:

### Step 1: Attack Generation
Generates 20,000 valid, stealthy stealth-checked FDIA vectors per grid network (IEEE 14-bus and IEEE 118-bus), scaling the Jacobian $H$ to match MW grid flow capacity and capping line reactances to prevent division-by-near-zero scaling issues.
```bash
python src/attack_generator.py
```
*Outputs generated:*
- `dataset/X_attacked_14.npy`, `dataset/X_attacked_118.npy`
- `dataset/gt_buses_14.npy`, `dataset/gt_buses_118.npy`
- `dataset/attack_types_14.json`, `dataset/attack_types_118.json`

---

### Step 2: Data Preprocessing
Partitions the datasets into 80/20 train/test stratified splits, fits standard scalers on the normal training splits to avoid feature scaling leakage, and saves the formatted partitions.
```bash
python src/preprocessing.py
```
*Outputs generated:*
- `dataset/X_train_14.npy`, `dataset/X_test_14.npy`, etc.
- `models/scaler_14.pkl`, `models/scaler_118.pkl`

---

### Step 3: Train FT-Transformer
Trains the optimized FT-Transformer models on GPU using batched training/evaluation configurations, Plateau Learning Rate schedulers, and Early Stopping. 
```bash
python src/ft_transformer.py
```
*Outputs generated:*
- `models/ft_transformer_14.pt`, `models/ft_transformer_118.pt`
- `dataset/y_pred_ft_14.npy`, `dataset/y_pred_ft_118.npy`

---

### Step 4: Train Detection Baselines
Trains baseline unsupervised models (LOF, OCSVM, PCA, GMM, COF, LUNAR) on the normal operating data, evaluates them on the test set, and generates comparative metrics.
```bash
python src/train.py
```
*Outputs generated:*
- `results/detection_14bus.csv`, `results/detection_118bus.csv`

---

### Step 5: Run SHAP Explainers
Performs SHAP Kernel Explanation on detected attack samples (`y_pred_ft == 1`) using a batched `predict_fn` logic to prevent memory overflows, generating beeswarm and bar feature importance summary plots.
```bash
python src/shap_analysis.py
```
*Outputs generated:*
- `outputs/shap_values_14.npy`, `outputs/shap_values_118.npy`
- `outputs/shap_bar_14bus.png`, `outputs/shap_beeswarm_14bus.png`, etc.

---

### Step 6: Attack Localization
Maps absolute line flow SHAP values back to their physical terminal buses using grid topology mapping, and evaluates localization accuracy (Top-1 and Top-3 accuracy) on true positive detected attacks.
```bash
python src/localization.py
```
*Outputs generated:*
- `results/localization_14bus.csv`, `results/localization_118bus.csv`

---

### Step 7: Compile Combined Table
Aggregates all detection and localization performance metrics into a single master summary table.
```bash
python src/generate_results_combined.py
```
*Outputs generated:*
- `results/results_combined.csv`

---

## File Verification

To verify that the entire workspace has completed successfully and all 39 output files exist, you can run:

```bash
python -c "
import os
for f in ['results/results_combined.csv', 'models/ft_transformer_118.pt', 'outputs/shap_beeswarm_118bus.png']:
    print(f, 'EXISTS' if os.path.exists(f) else 'MISSING')
"
```
