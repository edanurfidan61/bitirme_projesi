# Project Progress

## Overview

This thesis project ("bitirme_projesi") focuses on hyperspectral imaging and machine learning for medical plant stress detection and phytochemical prediction. It involves processing hyperspectral leaf images to detect stress and predict biochemical contents like chlorophyll and flavonols.

## Modules

### Module 0: Infrastructure
- **File**: load_envi.py
- **Status**: Completed
- **Purpose**: Load ENVI format hyperspectral images into Python numpy arrays. Handles .hdr metadata and .dat binary data to produce (512, 512, 204) float32 arrays.

### Module 1: Visualization
- **Files**: preprocessing.py, segmentation.py, visualize.py
- **Status**: Completed
- **Purpose**: Visualize hyperspectral data. Includes true-color RGB synthesis, leaf mask creation for background separation, and false-color index maps (e.g., ARI, NDVI).

### Module 2: Spectral Indices
- **File**: indices.py
- **Status**: Completed
- **Purpose**: Calculate spectral indices from hyperspectral reflectance data. Computes NDVI, GNDVI, ARI, RVSI, and ZTM indices for plant physiology assessment.

### Module 3
- **Status**: Not implemented

### Module 4: Feature Extraction
- **File**: features.py
- **Status**: Completed
- **Purpose**: Extract feature vectors from individual leaf images. Combines spectral band averages (204 dimensions) with index averages (5 dimensions) to create 209-dimensional feature vectors.

### Module 5: Dataset Creation
- **File**: dataset.py
- **Status**: Completed
- **Purpose**: Build a complete dataset from 204 leaf images. Loads ground truth from Dualex measurements, extracts features, and creates labeled datasets for chlorophyll, flavonol, and stress classification.

### Module 6: EDA and Modeling
- **Files**: eda.py, feature_engineering_v2.py, fix_fe_outputs.py, flav_deep_fe.py, pipeline_flav_v3.py, pipeline_flav_v5_pro.py, pipeline_models.py, utils_model.py, and subfolders with model files (e.g., model_gbc.py, model_rf_classify.py, etc.)
- **Status**: Completed (multiple versions and models trained)
- **Purpose**: Exploratory data analysis, advanced feature engineering, and machine learning pipelines. Includes regression models (GBR, PLSR, RF, SVR) and classification models (GBC, RF, SVM) for predicting chlorophyll, flavonols, and stress levels.

### Module 7
- **Status**: Not implemented

## Data Outputs
- Various dataset_output folders contain processed .npy files for features (X) and targets (y_chl, y_flav, y_stress).
- Model outputs in model_outputs folders include trained model results and comparisons.

## Next Steps
- Complete Module 3 and Module 7 if needed.
- Validate model performance and potentially deploy or further tune models.