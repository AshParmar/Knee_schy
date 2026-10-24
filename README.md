# Knee Osteoarthritis Severity Assessment: Complete End-to-End Neuro-Fuzzy Pipeline

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F79939?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An end-to-end deep learning framework for **Knee Osteoarthritis (OA) severity classification** from radiographic X-ray images (Kellgren-Lawrence Grades 0 to 4). 

This pipeline integrates a pre-trained **ResNet18 feature extraction backbone** with four interchangeable classification heads:
1. **ResNet18 + Linear Layer (Baseline)**
2. **ResNet18 + Takagi-Sugeno Fuzzy Classifier**
3. **ResNet18 + Spherical Fuzzy Classifier** ($\mu, \nu, \pi$ on a 3D unit sphere)
4. **ResNet18 + Neutrosophic Fuzzy Classifier** (Truth $T$, Indeterminacy $I$, Falsity $F$ decomposition)

---

## 🔄 Complete End-to-End Pipeline Architecture

```
                       END-TO-END PIPELINE WORKFLOW
                       
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: DATASET INGESTION                                                   │
│ Download Kaggle Knee Osteoarthritis Dataset via kagglehub                   │
│ Organize 5 Kellgren-Lawrence (KL) Severity Folders (Grades 0 to 4)          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2 & 3: DATA SPLITTING & MANAGEMENT                                     │
│ Load image paths & labels into Pandas DataFrames                            │
│ Stratified Split: Train Set, Validation Set, Test Set                       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4 & 5: DATA PIPELINE & AUGMENTATION                                    │
│ Resize to 224x224 | Augment: Random Horizontal Flip, Rotation, ColorJitter  │
│ PyTorch KneeOADataset & DataLoaders (Batch size: 32, Pin Memory, CUDA)     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: DEEP FEATURE EXTRACTION BACKBONE                                    │
│ Pre-trained ResNet18 (ImageNet weights)                                     │
│ Input: (B, 3, 224, 224) ──► Output: 512-Dimensional Deep Feature Vector      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 7: INTERCHANGEABLE EXPERIMENTAL CLASSIFICATION HEAD (MODEL_TYPE)       │
├─────────────────────────────────────────────────────────────────────────────┤
│  Option A: "linear"       ──► Standard Linear Head (W·x + b)                │
│  Option B: "takagi"       ──► Takagi-Sugeno Fuzzy Inference Layer           │
│  Option C: "spherical"    ──► Spherical Fuzzy Head (μ² + ν² + π² ≤ 1)       │
│  Option D: "neutrosophic" ──► Neutrosophic Head (T, I, F Decomposition)     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 8: OPTIMIZATION & TRAINING LOOP                                        │
│ CrossEntropyLoss | Adam Optimizer | Learning Rate Scheduler                 │
│ End-to-End Backpropagation across ResNet18 & Fuzzy Rule Parameters         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 9: COMPREHENSIVE EVALUATION & METRICS LOGGING                          │
│ Accuracy | Macro/Weighted Precision | Macro/Weighted Recall | F1-Score      │
│ Macro OVR ROC-AUC | Classification Report | Confusion Matrix Heatmap          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Detailed Pipeline Steps

### 🔹 Step 1: Environment Setup & Library Imports
* Imports core PyTorch libraries (`torch`, `torch.nn`, `torch.optim`), vision tools (`torchvision.transforms`, `torchvision.models`), evaluation suites (`sklearn.metrics`), dataset management (`kagglehub`, `pandas`, `PIL`), and visualizers (`matplotlib`, `seaborn`).
* Detects and configures hardware acceleration (`cuda` / `cpu`).

### 🔹 Step 2: Dataset Acquisition & Structure
* Automatically downloads the Knee Osteoarthritis X-ray dataset from Kaggle via `kagglehub`.
* Scans directory structures and builds structured DataFrames containing image file paths and target severity labels.
* **Classes (Kellgren-Lawrence Grading Scale)**:
  | KL Grade | Clinical Severity | Radiographic Description |
  | :---: | :---: | :--- |
  | **0** | Normal | No radiographic features of Osteoarthritis |
  | **1** | Doubtful | Doubtful joint space narrowing, minute osteophytes |
  | **2** | Minimal | Definite osteophytes, possible joint space narrowing |
  | **3** | Moderate | Moderate multiple osteophytes, definite joint space narrowing |
  | **4** | Severe | Severe joint space narrowing, bone-on-bone contact |

### 🔹 Step 3: Stratified Dataset Splitting
* Performs stratified train/validation/test splitting to maintain identical class distribution balances across all splits.

### 🔹 Step 4: Data Augmentation & Preprocessing
* **Training Transforms**:
  - Resize to $224 \times 224$ pixels.
  - `RandomHorizontalFlip(p=0.5)` for spatial invariance.
  - `RandomRotation(degrees=10)` for rotational invariance.
  - `ColorJitter(brightness=0.1, contrast=0.1)` for contrast variation robustness.
  - Conversion to Tensor and ImageNet normalization (Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]`).
* **Validation/Test Transforms**:
  - Resize to $224 \times 224$ pixels, Tensor conversion, and ImageNet normalization.

### 🔹 Step 5: Custom PyTorch Dataset Class (`KneeOADataset`)
* Custom PyTorch `Dataset` wrapper loading images on-demand using PIL, applying specified transformations, and returning `(image_tensor, label_tensor)` pairs.

### 🔹 Step 6: DataLoaders Creation
* Builds PyTorch `DataLoader` instances with configurable batch sizes (default: 32), shuffling for training data, and multi-threaded parallel data loading.

### 🔹 Step 7: Exploratory Data Visualization
* Plots sample X-ray images from each KL grade with overlay titles to visually inspect dataset quality and preprocessing transforms.

### 🔹 Step 8: Model Architecture & Classification Heads
Replaces ResNet18's default fully connected layer (`model.fc`) with one of four user-selectable classification heads using a single switch variable:

```python
MODEL_TYPE = "linear"       # ResNet18 + Linear Layer (Baseline)
MODEL_TYPE = "takagi"       # ResNet18 + Takagi-Sugeno Fuzzy Classifier
MODEL_TYPE = "spherical"    # ResNet18 + Spherical Fuzzy Classifier
MODEL_TYPE = "neutrosophic" # ResNet18 + Neutrosophic Fuzzy Classifier
```

#### Classification Heads Mathematical Breakdown:

1. **Baseline Linear Layer (`nn.Linear`)**:
   $$\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$$
   Standard affine linear mapping from 512 features to 5 class logits.

2. **Takagi-Sugeno Fuzzy Classifier (`TakagiSugenoFuzzyLayer`)**:
   - **Gaussian Antecedents**: $\mu_{r,d}(x_d) = \exp\left(-\frac{(x_d - c_{r,d})^2}{2\sigma_{r,d}^2}\right)$
   - **Log-Space Rule Firing**: $S_r(x) = -\frac{1}{2} \sum_{d=1}^{512} \left(\frac{x_d - c_{r,d}}{\sigma_{r,d}}\right)^2 \cdot \frac{1}{\sqrt{512}}$
   - **Softmax Normalization**: $\bar{w}_r(x) = \frac{\exp(S_r(x))}{\sum_k \exp(S_k(x))}$
   - **Consequent Blending**: $\text{Logits}(x) = \sum_{r=1}^{R} \bar{w}_r(x) \cdot \mathbf{a}_r$

3. **Spherical Fuzzy Classifier (`SphericalFuzzyClassifier`)**:
   Maps features onto a 3D unit sphere enforcing $\mu^2 + \nu^2 + \pi^2 \le 1$:
   - **Spherical Coordinates**:
     $$r_r(x) = \exp(-0.5 d_r(x)) \in (0, 1]$$
     $$\theta_r(x) = \frac{\pi}{2} \sigma(w_\theta d_r + b_\theta), \quad \phi_r(x) = \frac{\pi}{2} \sigma(w_\phi d_r + b_\phi)$$
     $$\mu_r = r_r \sin\theta_r \cos\phi_r, \quad \nu_r = r_r \sin\theta_r \sin\phi_r, \quad \pi_r = r_r \cos\theta_r$$
   - **Spherical Score Function**: $\text{Score}_r = \mu_r^2 - \nu_r^2 + \gamma \mu_r (1 - \pi_r)$
   - **Firing & Blending**: $\bar{w}_r = \text{Softmax}(\text{Score}_r)$, $\text{Logits}(x) = \sum_r \bar{w}_r \mathbf{a}_r$

4. **Neutrosophic Fuzzy Classifier (`NeutrosophicClassifier`)**:
   Decouples input features into independent Truth ($T$), Indeterminacy ($I$), and Falsity ($F$) components ($0 \le T + I + F \le 3$):
   - **Rule Triplet**: $T_r(x) = \exp(-0.5 d_r(x))$, $I_r(x) = \sigma(w_I d_r + b_I)$, $F_r(x) = 1.0 - T_r(x)$
   - **Rule Score**: $\text{Score}_r = T_r(1 - I_r) - F_r + \alpha(T_r - F_r)$
   - **Direct Projections**: $T_c = \sigma(\mathbf{W}_T \mathbf{x} + \mathbf{b}_T)$, $I_c = \sigma(\mathbf{W}_I \mathbf{x} + \mathbf{b}_I)$, $F_c = \sigma(\mathbf{W}_F \mathbf{x} + \mathbf{b}_F)$
   - **Final Fusion**: $\text{Logits}(x) = \text{MatMul}(\text{Softmax}(\text{Score}_r), \mathbf{a}_r) + (T_c - F_c - \beta I_c)$

### 🔹 Step 9: Loss Function, Optimizer & Learning Rate Scheduler
* **Loss Function**: `nn.CrossEntropyLoss()` for multi-class classification.
* **Optimizer**: `optim.Adam(model.parameters(), lr=0.001)`. Updates both ResNet backbone weights and fuzzy layer parameters (centers, sigmas, angular weights, consequents).
* **Learning Rate Scheduler**: Adjusts learning rate dynamically per epoch to ensure steady convergence.

### 🔹 Step 10: Training & Validation Loop
* Iterates through training epochs.
* Executes forward pass, calculates loss, performs backward pass (`loss.backward()`), and updates model weights (`optimizer.step()`).
* Tracks training and validation loss and accuracy per epoch.

### 🔹 Step 11: Comprehensive Model Evaluation
Evaluates the trained model on the holdout test set to compute:
* **Accuracy**: Overall classification accuracy.
* **Precision (Macro & Weighted)**: Measure of positive predictive value.
* **Recall (Macro & Weighted)**: Measure of sensitivity.
* **F1-Score (Macro & Weighted)**: Harmonic mean of precision and recall.
* **ROC-AUC (Macro One-vs-Rest)**: Area under ROC curve across multi-class probabilities.
* **Classification Report**: Per-class precision, recall, and F1-score breakdown.
* **Confusion Matrix Heatmap**: Visual matrix plot showing true vs. predicted grades.

### 🔹 Step 12: Architectural Comparison & Medical Discussion
Provides a comparative table and clinical breakdown explaining why each model behaves differently when classifying continuous Osteoarthritis severity grades.

---

## ⚖️ Model Comparison Summary Table

| Model Architecture | Mathematical Foundation | Parameter Constraints | Clinical Advantage in Medical Imaging |
| :--- | :--- | :--- | :--- |
| **1. Linear (Baseline)** | Affine projection $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$ | Unconstrained weights $\mathbf{W} \in \mathbb{R}^{5 \times 512}$ | Standard benchmark baseline; prone to overconfidence on continuous boundaries. |
| **2. Takagi-Sugeno** | Gaussian antecedents + linear consequents | Centers $c_r$, widths $\sigma_r$, consequents $\mathbf{a}_r$ | Soft membership matching for smooth interpolation between KL grades. |
| **3. Spherical Fuzzy** | 3D Unit sphere $(\mu, \nu, \pi)$ | Spherical coordinates $(r, \theta, \phi)$ with $\mu^2 + \nu^2 + \pi^2 \le 1$ | Explicitly quantifies diagnostic hesitation $\pi$ in ambiguous joint space narrowing. |
| **4. Neutrosophic** | Triplet decomposition $(T, I, F)$ | Independent $T, I, F \in [0, 1]$ satisfying $0 \le T + I + F \le 3$ | Explicitly isolates radiographic image noise ($I$) from diagnostic evidence ($T$). |

---

## 📁 Repository Files

```
.
├── knee.ipynb      # Complete executable Jupyter Notebook containing all 12 pipeline steps
└── README.md       # Project Documentation & Pipeline Overview
```

---

## ⚙️ Installation & Requirements

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn pillow tqdm kagglehub
```

### Hardware Acceleration
The pipeline automatically utilizes GPU acceleration if available:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

---

## 🚀 How to Run the Pipeline

1. Open `knee.ipynb` in VS Code, Jupyter Lab, or Google Colab.
2. Go to **Step 8** in the notebook and set your desired `MODEL_TYPE`:
   ```python
   MODEL_TYPE = "linear"       # or "takagi", "spherical", "neutrosophic"
   ```
3. Run all notebook cells sequentially. The notebook will handle dataset downloading, preprocessing, training, evaluation, and plotting automatically.

---

## 📄 License
Distributed under the MIT License.
