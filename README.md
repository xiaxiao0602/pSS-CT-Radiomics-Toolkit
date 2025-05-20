# pSS‑CT‑Radiomics‑Toolkit

> A complete end-to-end toolkit for non‑invasive, interpretable early diagnosis of primary Sjögren’s syndrome (pSS) using CT images.

---

## 🔍 Features

1. **Automatic Segmentation**  
   - nnUNetv2‑based accurate delineation of salivary glands (parotid & submandibular)

2. **Feature Extraction & Enhancement**  
   - PyRadiomics for 1,409 texture features  
   - Variational autoencoder (CVAE)–based data augmentation

3. **Model Training**  
   - Algorithms: XGBoost, Random Forest, LightGBM  
   - Ensemble strategies: Voting & Stacking  

4. **Interpretability Analysis**  
   - SHAP‑based global & individual feature importance  

5. **Feature Visualization**  
   - 3D voxel‑level rendering  
   - Interactive visualization scripts

6. **Web Prediction Platform**  
   - Flask server for DICOM/NIfTI upload  
   - Automatic segmentation, probability output & SHAP force plots  

---

## 🚀 Quick Start

