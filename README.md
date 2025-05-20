# pSS-CT-Radiomics-Toolkit
### Our warehouse provides a whole process toolkit for early diagnosis of primary Sjogren's syndrome (pSS) based on CT images, covering:
### 
Automatic segmentation: nnUNetv2 based salivary gland (parotid gland, submandibular gland) accurate segmentation script

Feature extraction and enhancement: PyRadiomics was used to extract 1,409 features, and data enhancement was implemented based on CVAE

Model training: Training and verification of XGBoost, Random Forest, LightGBM and other algorithms and voting/stacking integration strategies

Interpretability analysis: SHAP based global/individual feature importance interpretation and visualization tool

Feature visualization: 3D voxel-level feature rendering and interactive display script

Web prediction platform: Flask built online upload DICOM/NIfTI and automatically output segmentation results, prediction probability and SHAP force map

### With this project, you can quickly replicate the end-to-end pipeline described in this paper and perform fine‑tune and secondary development on your own datasets to facilitate noninvasive, interpretable early diagnosis of pSS.
