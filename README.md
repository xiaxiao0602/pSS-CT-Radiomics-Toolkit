\section*{pSS-CT-Radiomics-Toolkit}
\textit{A comprehensive end-to-end toolkit for early diagnosis of primary Sjögren’s syndrome (pSS) based on CT images.}

\subsection*{Key Components}
\begin{description}[leftmargin=!,labelwidth=\widthof{\bfseries Feature visualization:}]
  \item[Automatic segmentation:]  
    nnUNetv2-based scripts for precise segmentation of salivary glands (parotid and submandibular).
  
  \item[Feature extraction \& enhancement:]  
    Extraction of 1,409 radiomic features via PyRadiomics, with data augmentation powered by a conditional variational autoencoder (CVAE).
  
  \item[Model training:]  
    Training and validation pipelines for XGBoost, Random Forest, LightGBM, and ensemble strategies (voting \& stacking).
  
  \item[Interpretability analysis:]  
    SHAP-based tools for global and patient‑level feature importance interpretation and visualization.
  
  \item[Feature visualization:]  
    3D voxel‑level feature rendering and interactive display scripts for in-depth exploration.
  
  \item[Web prediction platform:]  
    Flask application enabling online upload of DICOM/NIfTI files, with automatic segmentation, probability prediction, and SHAP force‑plot generation.
\end{description}

\subsection*{Getting Started}
Clone the repository and install dependencies:
\begin{verbatim}
git clone https://github.com/yourusername/pSS-CT-Radiomics-Toolkit.git
cd pSS-CT-Radiomics-Toolkit
pip install -r requirements.txt
\end{verbatim}

\subsection*{Usage Examples}
\begin{itemize}
  \item \textbf{Segmentation:}
    \begin{verbatim}
    python scripts/segment.py \
      --input path/to/dicom_folder \
      --output path/to/segmentation.nii.gz
    \end{verbatim}
  
  \item \textbf{Feature Extraction:}
    \begin{verbatim}
    python scripts/extract_features.py \
      --image path/to/ct.nii.gz \
      --mask path/to/segmentation.nii.gz \
      --out features.csv
    \end{verbatim}
  
  \item \textbf{Web Deployment:}
    \begin{verbatim}
    cd web_app
    flask run --host=0.0.0.0 --port=5000
    \end{verbatim}
\end{itemize}

\subsection*{Contributing}
Contributions are welcome! Please submit issues and pull requests on GitHub.

\subsection*{License}
This project is licensed under the MIT License. See \texttt{LICENSE} for details.

