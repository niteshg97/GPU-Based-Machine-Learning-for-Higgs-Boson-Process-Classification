# ⚛️ GPU-Based Machine Learning for Higgs Boson Process Classification

## 🧠 Introduction
The discovery of the **Higgs boson particle** at CERN in 2012 marked a monumental achievement in modern physics, confirming the existence of the **Higgs field** — a cornerstone of the Standard Model.  
Detecting such elusive particles requires **advanced computing**, sophisticated **machine learning (ML)**, and global collaboration 🌍.

This repository implements a **GPU-accelerated machine learning pipeline** for classifying **signal processes** (Higgs-producing) vs **background processes** (non-Higgs).  
It leverages the **RAPIDS** framework for GPU-parallel data processing and **Google Colab (NVIDIA T4 GPU)** for accelerated model training, evaluation, and testing. ⚡

---

## 🧩 Background

### 🧬 Machine Learning in High-Energy Physics
ML has become an essential tool in **particle physics**, traditionally applied to processed data from reconstruction algorithms.  
Today’s approaches enable **direct analysis of raw detector data**, helping with:
- **Event selection**
- **Event classification**
- **Background suppression**

These innovations allow physicists to separate meaningful *signal* events from enormous *background* noise.

---

### ⚙️ Computational Demands
High-energy physics experiments — like those at the **Large Hadron Collider (LHC)** — generate **tens of terabytes per second** of raw data.  
The **High Luminosity LHC (HL-LHC)** will produce up to **15× more data**.  
Traditional CPUs struggle with this scale, motivating the use of **GPUs** for their massive parallelism in matrix operations and data transformations 🚀.

---

### 🎯 Task Specification
In collider experiments, **signal events** correspond to Higgs boson decay, while **background events** come from other particles.  
Using **Monte Carlo–simulated data**, this project applies machine learning to classify events as either **signal (1)** or **background (0)**.

---

## 🧱 Implementation Steps

### 🗂️ Data
- **Dataset:** [UCI HIGGS Dataset](https://archive.ics.uci.edu/ml/datasets/HIGGS)  
- **Size:** ~11 million instances generated via Monte Carlo simulations.  
- **Features:** 29 total — first column = binary label (1 = signal, 0 = background), 28 physics-based features.  

---

### 🔍 Exploratory Data Analysis (EDA)
Key analyses performed:
- Checked for **missing values** and **class imbalance**.
- Visualized **feature correlations** to identify redundancy.
- Identified and managed **outliers** using the **3×IQR rule** (less aggressive, preserving more data).  
- Noted strong correlation pairs like `m_wbb` ↔ `m_wwbb` and `m_jj` ↔ `m_jjj`.

✅ **Feature reduction:** Removed `m_wbb` and `m_jjj` to prevent redundancy.

---

### 🧹 Data Preprocessing
- **Removed correlated features** (to avoid multicollinearity).  
- **Handled outliers** (3×IQR trimming).  
- **Addressed class imbalance** using **SMOTE (Synthetic Minority Over-sampling Technique)**.  
- **Normalized** using `cuML`’s GPU-based standard scaler for better convergence and accuracy.  
- Saved processed datasets for reuse and rapid model training.

---

## 🤖 Models
Three machine learning models were trained and compared:

| 🔢 Model | ⚙️ Description | 🚀 Accelerator |
|:--|:--|:--|
| **Logistic Regression** | Linear baseline classifier | GPU/CPU |
| **Random Forest** | Ensemble model with bootstrapped decision trees | GPU/CPU |
| **XGBoost** | Gradient-boosted trees (best-performing) | GPU (`gpu_hist`) |

---

## 🧪 Experiments
### 🧮 Grid Search & Tuning
A **grid search** was used for hyperparameter tuning to achieve optimal performance.

- **Logistic Regression:** Tested on CPU & GPU — similar accuracy, but **GPU 3× faster** ⏩.  
- **Random Forest:** Tuned `n_estimators` and `max_depth`; balanced accuracy and computational load.  
- **XGBoost:** Trained with and without normalization — **normalization significantly improved stability and AUC**.

### 🧭 Principal Component Analysis (PCA)
- Applied **PCA** to explore dimensionality reduction.  
- Re-trained models on reduced feature sets.  
- PCA preserved ~95% variance with 18 components — slight drop in accuracy but faster training time.  
- **XGBoost maintained top performance even with reduced dimensions.**

---

## 🧪 Testing and Model Selection
After multiple experiments, **Logistic Regression** was dropped due to underperformance.

| 🧠 Model | Accuracy | Precision | Recall | Specificity | F1-Score | AUC | Training Time (s) |
|:--------:|:---------:|:----------:|:----------:|:----------:|:--------:|:----:|:----------------:|
| **Logistic Regression** | 0.61 | 0.61 | 0.61 | 0.62 | 0.69 | 0.73 | 52 |
| **Random Forest** | 0.73 | 0.74 | 0.72 | 0.73 | 0.73 | 0.80 | 176 |
| **🌟 XGBoost (Selected)** | **0.74** | **0.75** | **0.73** | **0.74** | **0.74** | **0.82** | **132** |

✅ **XGBoost outperformed all models**, particularly in **precision and AUC**, making it the most suitable model for Higgs boson process classification.

📦 The trained XGBoost model is saved in:


------

## 🧭 Conclusion
This project demonstrates the effectiveness of **GPU-based ML** in high-energy physics for **Higgs boson process classification**.

**Key achievements:**
- ⚡ **Accelerated training** with Google Colab (T4 GPU).  
- 📊 Efficient handling of an **11M-row dataset** using RAPIDS-based and GPU-enhanced libraries.  
- 🧩 Rigorous preprocessing pipeline (outlier handling, normalization, feature reduction).  
- 🏆 **XGBoost achieved the best accuracy (74%) and AUC (0.82)** — recommended model for deployment.  

---

## ⚙️ Tools & Libraries
- **Python 3.10**
- **Google Colab (NVIDIA T4 GPU)**
- **RAPIDS cuML/cuDF**
- **XGBoost** (GPU `gpu_hist` mode)
- **scikit-learn**
- **pandas**, **matplotlib**, **seaborn**
- **imbalanced-learn (SMOTE)**

---

## 📚 References  
1. UCI Machine Learning Repository — [HIGGS Dataset](https://archive.ics.uci.edu/ml/datasets/HIGGS).  
2. RAPIDS Framework — [https://rapids.ai](https://rapids.ai).  
3. XGBoost Documentation — [https://xgboost.readthedocs.io](https://xgboost.readthedocs.io).  
4. CERN Open Data Portal — [https://opendata.cern.ch](https://opendata.cern.ch).  

---

## 🏁 Summary
> **Final Recommendation:**  
> Use **XGBoost (GPU-accelerated)** as the primary classifier for Higgs boson process discrimination.  
> Achieved **74% accuracy** and **AUC = 0.82** on the test dataset — with superior inference speed and stability on GPU compared to CPU models. ⚡  

**“GPU computing has transformed theoretical physics from hours to minutes — accelerating discovery itself.”**

