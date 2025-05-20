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
   - Publication‑quality plots for insight and reporting

5. **Feature Visualization**  
   - 3D voxel‑level rendering  
   - Interactive visualization scripts

6. **Web Prediction Platform**  
   - Flask server for DICOM/NIfTI upload  
   - Automatic segmentation, probability output & SHAP force plots  

---

## 🚀 Quick Start

```bash
git clone https://github.com/your_username/pSS-CT-Radiomics-Toolkit.git
cd pSS-CT-Radiomics-Toolkit

# Create environment
conda env create -f environment.yml
conda activate pss-ct-toolkit

# Train a model
python train.py --config configs/xgb.yaml

# Launch the prediction web app
cd web_app
python app.py
