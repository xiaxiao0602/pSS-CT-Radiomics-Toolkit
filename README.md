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

```bash
# 1. Clone the repository
git clone https://github.com/your_username/pSS-CT-Radiomics-Toolkit.git
cd pSS-CT-Radiomics-Toolkit

# 2. Create & activate the conda environment
conda env create -f environment.yml
conda activate pss-ct-toolkit

# 3. Prepare your data
# Place your DICOM/NIfTI files under `data/` or adjust the --input path accordingly
mkdir -p data/input
# e.g. copy your scans:
cp /path/to/scans/*.nii.gz data/input/

# 4. Train or evaluate prediction models
# (in the "Sjogren's Syndrome Prediction Models" folder)
python "Sjogren's Syndrome Prediction Models/train.py" \
  --config "Sjogren's Syndrome Prediction Models/configs/xgb.yaml" \
  --data-dir data/input \
  --output-dir models/

# 5. Run the feature‐visualization demo
# (in the "Feature Visualization Application" folder)
cd "Feature Visualization Application"
python app.py --model ../models/best_model.pkl --features ../radiomics/features.csv

# 6. Launch the web prediction platform
# (in the "AutoSS Website" folder)
cd ../"AutoSS Website"
export FLASK_APP=app.py
flask run --host=0.0.0.0 --port=5000

# Then open your browser at http://localhost:5000

