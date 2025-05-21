# 3D Feature Map Visualizer

This application is designed for visualizing 3D feature maps and ROI (Region of Interest) masks in medical imaging, and provides reference example comparison functionality.

## Features

- Load and visualize 3D feature maps and ROI masks  
- Select different features for visualization via dropdown menu  
- Adjust feature map opacity (0.1–0.9)  
- Provide negative and positive reference examples for comparison  
- Interactive 3D rendering with support for rotation, zooming, and panning  
- Intelligent caching mechanism for faster reference example loading  
- Color map legend display  
- Coordinate axis labeling (unit: mm)

## Installation Instructions

This program includes a standalone Python environment and can run without requiring Python to be installed on your system. For first-time use, please follow the steps below:

1. Download and unzip the program package  
2. Double-click `setup.bat` and wait for the environment setup to complete  
   - This process will download a portable version of Python 3.9.13  
   - It will automatically install all required dependencies  
   - Initial setup may take a few minutes

## Running the Program

After installation is complete, launch the program using the following steps:

1. Double-click `start.bat` to launch the program  
   - No need to configure any environment variables  
   - All dependencies are contained within the program directory

## User Guide

1. After launching the program, click the **"Select Feature Map Folder"** button to choose a folder containing the feature maps and ROI mask:  
   - The folder must contain a file named `mask.nii.gz` as the ROI mask  
   - Other `.nii` or `.nii.gz` files will be treated as feature maps

2. Use the control panel on the left:  
   - Select the feature to visualize from the dropdown menu  
   - Adjust the opacity dropdown to change feature map visibility  
   - Review the feature prediction direction guide to understand predictive tendencies  
   - View negative and positive reference examples for comparative analysis

3. In the 3D view window:  
   - Rotate the model with the left mouse button  
   - Zoom with the right mouse button or scroll wheel  
   - Pan the view with the middle mouse button

## Reference Example Setup

The program requires a `reference_examples` folder in the run directory with the following structure:

```
reference_examples/
├── negative/
│   ├── feature1.nii.gz
│   ├── feature2.nii.gz
│   └── mask.nii.gz
└── positive/
    ├── feature1.nii.gz
    ├── feature2.nii.gz
    └── mask.nii.gz
```


- The `negative` folder stores negative reference examples  
- The `positive` folder stores positive reference examples  
- Each folder must include the corresponding `mask.nii.gz` file  
- Feature file names must match the feature names shown in the main window

## Feature Prediction Direction Guide

- **Features where higher values indicate a negative prediction:**
  - `square_glcm_MaximumProbability`

- **Features where higher values indicate a positive prediction:**
  - `square_glcm_DifferenceAverage`  
  - `square_glrlm_RunPercentage`  
  - `square_glrlm_ShortRunEmphasis`

## File Requirements

- **Feature map files**: must be in `.nii` or `.nii.gz` format  
- **ROI mask file**: must be named `mask.nii.gz`  
- All files must have the same dimensions and spatial alignment  
- Reference example files must follow the same naming rules as the main feature files

## Troubleshooting

If the program fails to start or encounters errors, please check the following:

1. Ensure that `setup.bat` was run to complete the environment setup  
2. Verify that the folder structure is correct  
3. Ensure reference example filenames match the feature filenames  
4. Check that all file formats are correct

If the issue persists, try re-running `setup.bat` to reset the environment.
