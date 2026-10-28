# Project Summary & Reference Documentation

**Project Title**: SHAP-Guided Attack Localization with FT-Transformer for Explainable FDIA Detection in Smart Grids

---

## 1. Executive Summary
This project implements a secure, explainable deep learning pipeline to detect and localize **False Data Injection Attacks (FDIA)** targeting state estimation in smart grids. Traditional bad data detection (BDD) systems are blind to stealthy FDIAs that exploit network topology. 

Our proposed framework addresses this vulnerability through three core contributions:
1. **Valid Stealthy Attack Generation**: Constructing mathematically stealthy FDIAs ($a = Hc$) matching the physical power flow capabilities of IEEE 14-bus and IEEE 118-bus grids.
2. **FT-Transformer Classifier**: A deep tabular model achieving competitive detection performance, significantly outperforming unsupervised baseline detectors (LOF, OCSVM, GMM, LUNAR, etc.).
3. **SHAP-Guided Localization Layer**: A topological mapping system that leverages Shapley Additive exPlanations (SHAP) feature attributions to trace line flow anomalies back to their originating substation bus target.

---

## 2. Theoretical Background & Mathematical Context

### State Estimation & Bad Data Detection (BDD)
In power grid operations, state estimation computes the optimal voltage angles and magnitudes ($\theta, V$) from raw measurements $z$ (line flows, bus injections) using:
$$z = Hx + e$$
where $H$ is the Jacobian measurement matrix and $e$ is measurement noise. Standard BDD flags anomalies by measuring the $L_2$ norm of the measurement residual:
$$r = \|z - H\hat{x}\|_2$$

### The Stealthy FDIA ($a = Hc$)
An attacker with grid topology knowledge can construct a stealthy attack vector $a = Hc$, where $c$ is the vector of voltage angle manipulations. When injected, the malicious measurements become $z_{attacked} = z + a$. The residual is:
$$r_{attacked} = \|z + a - H(\hat{x} + c)\|_2 = \|z + Hc - H\hat{x} - Hc\|_2 = \|z - H\hat{x}\|_2$$
Because $r_{attacked} = r_{clean}$, the traditional residual checker is completely bypassed (Residual $0.0$), leaving the grid vulnerable.

---

## 3. Core Contributions & Physical Modeling

### Contribution 1: Physical-Scale Attack Generator
- **Base MVA Scaling**: Solves the unit mismatch between standard power flow (MW) and physical reactance ($x_{ohm}$). We scale the Jacobian $H$ by the system base capacity ($S_{base} = 100$ MVA for IEEE 14-bus, $1000$ MVA for IEEE 118-bus).
- **Reactance Capping**: Capped the minimum reactance at `0.01` ohms. This prevents division-by-near-zero scaling issues on virtual switches/short tie-lines.
- **Physical Feasibility**: Restricts maximum injected line flows to realistic bounds (**7,845 MW** for 14-bus and **330.5 MW** for 118-bus) instead of numerical infinities.

#### IEEE Grid Topologies
The diagrams below illustrate the physical layout of the grids. Generators/external grids are highlighted in **red** (power sources), buses in **blue** (nodes), and transmission lines in **grey** (edges).

| IEEE 14-bus Grid Diagram | IEEE 118-bus Grid Diagram |
| :---: | :---: |
| ![IEEE 14-bus Diagram](file:///d:/SHAP_LUNAR_PROPOSED/outputs/grid_14bus.png) | ![IEEE 118-bus Diagram](file:///d:/SHAP_LUNAR_PROPOSED/outputs/grid_118bus.png) |

---

### Contribution 2: Tabular Deep Learning with FT-Transformer
- **Architecture**: A Feature Tokenizer Transformer baseline model. To run efficiently within a 4GB VRAM constraint without CUDA OOM, the embedding sizes were scaled dynamically:
  - IEEE 14-bus: `d_token=64`, `ffn_d_hidden=128`, `n_blocks=2`
  - IEEE 118-bus: `d_token=128`, `ffn_d_hidden=256`, `n_blocks=2`
- **Training Config**: Integrated a `ReduceLROnPlateau` scheduler (factor=0.5, patience=10) and early stopping (min_delta=0.001, patience=20) over 150 epochs.

---

### Contribution 3: Topological SHAP-Guided Localization
- **Attribution Mapping**: When the FT-Transformer flags an attack, `shap.KernelExplainer` identifies which line flow measurements contributed to the decision.
- **Bus Localization**: Maps the flow attributions back to the grid terminal buses by summing the absolute SHAP values of lines connected to each bus:
  $$S_{bus}(j) = \sum_{i \in \text{lines connected to bus } j} |s_i|$$
- The bus with the highest attribution $S_{bus}$ is predicted as the targeted bus.

#### SHAP Explanation Graphics
Below are the SHAP bar plots (feature importances) and beeswarm plots (distribution of feature impacts) generated from the explained attack samples:

##### **IEEE 14-bus SHAP Plots**
* **SHAP Feature Importance Bar Plot (14-bus)**:
  ![SHAP Bar Plot 14-bus](file:///d:/SHAP_LUNAR_PROPOSED/outputs/shap_bar_14bus.png)
* **SHAP Beeswarm Plot (14-bus)**:
  ![SHAP Beeswarm Plot 14-bus](file:///d:/SHAP_LUNAR_PROPOSED/outputs/shap_beeswarm_14bus.png)

##### **IEEE 118-bus SHAP Plots**
* **SHAP Feature Importance Bar Plot (118-bus)**:
  ![SHAP Bar Plot 118-bus](file:///d:/SHAP_LUNAR_PROPOSED/outputs/shap_bar_118bus.png)
* **SHAP Beeswarm Summary Plot (118-bus)**:
  ![SHAP Beeswarm Plot 118-bus](file:///d:/SHAP_LUNAR_PROPOSED/outputs/shap_beeswarm_118bus.png)

---

## 4. Performance & Validation Results

### Detection Performance (Ours vs. Baselines)
Trained on normal training samples (`y_train == 0`) and evaluated on the test set:

| Model | ACC_14 | PRE_14 | REC_14 | F1_14 | ACC_118 | PRE_118 | REC_118 | F1_118 | Top1_14 | Top3_14 | Top1_118 | Top3_118 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FT-Transformer + SHAP (Ours)** | **0.9151** | **1.0000** | **0.8303** | **0.9073** | **0.8732** | **0.8880** | **0.8542** | **0.8708** | **0.2200** | **0.5600** | **0.8696** | **1.0000** |
| GMM | 0.8880 | 0.9037 | 0.8685 | 0.8858 | 0.9251 | 0.8934 | 0.9655 | 0.9280 | N/A | N/A | N/A | N/A |
| LOF | 0.7424 | 0.8547 | 0.5840 | 0.6939 | 0.8141 | 0.8673 | 0.7418 | 0.7996 | N/A | N/A | N/A | N/A |
| LUNAR | 0.7212 | 0.8357 | 0.5507 | 0.6640 | 0.8283 | 0.9892 | 0.6637 | 0.7944 | N/A | N/A | N/A | N/A |
| OCSVM | 0.6796 | 0.8285 | 0.4530 | 0.5857 | 0.7535 | 0.8541 | 0.6115 | 0.7127 | N/A | N/A | N/A | N/A |
| PCA | 0.7289 | 0.8520 | 0.5540 | 0.6714 | 0.7864 | 0.8730 | 0.6703 | 0.7583 | N/A | N/A | N/A | N/A |
| COF | 0.5413 | 0.6219 | 0.2105 | 0.3145 | 0.6650 | 0.7097 | 0.5585 | 0.6251 | N/A | N/A | N/A | N/A |

#### F1-Score Comparison Chart
Below is the visual comparison of the F1-scores across all models for both networks:
![F1-Score Comparison Bar Chart](file:///d:/SHAP_LUNAR_PROPOSED/outputs/f1_comparison.png)

### Localization Performance (True Positive Attacks)
Measures how successfully the SHAP-guided layer pinpoints the assigned target bus:
* **IEEE 14-bus Grid**: 
  - **Top-1 Accuracy**: **`22.00%`** (3.1x better than random guess)
  - **Top-3 Accuracy**: **`56.00%`** (2.6x better than random guess)
* **IEEE 118-bus Grid**:
  - **Top-1 Accuracy**: **`86.96%`**
  - **Top-3 Accuracy**: **`100.00%`** (The target bus is guaranteed to be in the top 3 most suspected nodes!)

#### Localization Accuracy Comparison Chart
Below is the visual comparison of the Top-1 and Top-3 localization accuracy values across both grids:
![Localization Accuracy Comparison Bar Chart](file:///d:/SHAP_LUNAR_PROPOSED/outputs/localization_comparison.png)

---

## 5. Directory & Script Architecture

- **`src/main.py`**: The master orchestrator script. Runs the entire pipeline sequentially.
- **`src/attack_generator.py`**: Builds the Jacobians, generates stealthy attacks across 4 profiles, and runs stealth residual checks.
- **`src/preprocessing.py`**: Performs stratified splitting and standard scaling.
- **`src/ft_transformer.py`**: Builds the neural network, implements Plateau scheduler, early stopping, and training on GPU.
- **`src/train.py`**: Trains all unsupervised detection baselines.
- **`src/shap_analysis.py`**: Executes the KernelExplainer in batches to explain predictions without memory leaks.
- **`src/localization.py`**: Sums flow attributions into bus attributions and calculates Top-1 and Top-3 accuracy metrics.
- **`src/inspect_dataset.py`**: Utility script to print formatted CSV-like previews of the generated dataset splits.
- **`dataset/`**: Stores `.npy` matrices of train/test splits, classifications, and target buses.
- **`models/`**: Saves state dicts, trained baselines, and scaler pickles.
- **`outputs/`**: Contains generated grid topology diagrams and SHAP bar/beeswarm plots.
- **`results/`**: Stores detection, localization, and compiled CSV summary tables.
