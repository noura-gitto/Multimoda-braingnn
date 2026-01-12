# CatBoost Code and Results Review Report

## 1. Introduction

This report provides a detailed review of the CatBoost implementation within the `Multimoda-braingnn` repository, focusing on its core code files and the experimental results generated. The CatBoost model serves as a baseline or alternative approach for Autism Spectrum Disorder (ASD) classification using multimodal neuroimaging data (fMRI and sMRI) and phenotypic information.

## 2. Code Review

### 2.1. `main.py`: Orchestration of the CatBoost Pipeline

The `main.py` script acts as the central orchestrator for the CatBoost classification pipeline. Its primary responsibilities include:

*   **Reproducibility**: Sets random seeds to ensure consistent results across runs.
*   **Data Handling**: Coordinates the loading of fMRI, sMRI, and phenotypic data using functions from `data_loader.py`.
*   **Cross-Validation**: Implements a stratified K-fold cross-validation strategy, specifically designed to be 
site-aware to handle the heterogeneity of multi-site neuroimaging datasets. This ensures that each fold maintains a representative distribution of sites and labels.
*   **ComBat Harmonization**: Integrates `apply_combat_per_fold` to perform ComBat harmonization, a critical step for removing site-specific batch effects from fMRI and sMRI data. Importantly, this harmonization is applied using only training data parameters to prevent data leakage into validation and test sets.
*   **Feature Engineering**: Utilizes `prepare_combined_features` to perform feature selection and scaling. This involves applying Recursive Feature Elimination (RFE) with a RidgeClassifier estimator to both fMRI and sMRI data independently, and then on their combined features. Scalers and selectors are fitted exclusively on training data.
*   **Model Training**: Calls `train_catboost` from `model_utils.py` to train the CatBoost classifier, including hyperparameter search for learning rate and tree depth.
*   **Evaluation and Visualization**: After training, it evaluates the model using `evaluate_model` and generates various visualizations (ROC curves, confusion matrices, feature importance, etc.) using functions from `visualization.py`.

### 2.2. `config.py`: Centralized Configuration Management

The `config.py` file centralizes all configurable parameters for the CatBoost pipeline, promoting code maintainability and ease of experimentation. Key configurations include:

*   **Paths**: Defines base directories for data, labels, sMRI, and model saving, including dynamic path generation based on harmonization settings.
*   **ComBat Settings**: Flags (`COMBAT_FMRI`, `COMBAT_SMRI`) to enable or disable ComBat harmonization for each modality.
*   **Data Parameters**: Specifies the number of folds (`K_FOLD`), total number of samples (`NUM_SAMPLES`), and a list of `USELESS_SAMPLES` to be excluded due to missing or corrupted data.
*   **Feature Selection Parameters**: Defines the target number of features (`NEW_FEATURES_FMRI`, `NEW_FEATURES_SMRI`, `NEW_FEATURES_COMBINE`) after RFE, and the step sizes for RFE (`RFE_STEP_FMRI`, `RFE_STEP_SMRI`).
*   **CatBoost Hyperparameters**: Lists possible `LEARNING_RATES` and `DEPTHS` for hyperparameter tuning, along with `ITERATIONS`.
*   **Random Seed**: Ensures reproducibility with a fixed `RANDOM_SEED`.
*   **Visualization Settings**: Parameters for figure DPI and size.

### 2.3. `model_utils.py`: Core Model Utilities

The `model_utils.py` script encapsulates essential functions related to feature selection, CatBoost training, and model persistence.

*   **`feature_selection_fmri` and `feature_selection_smri`**: These functions implement Recursive Feature Elimination (RFE) using a `RidgeClassifier` as the estimator. They are crucial for dimensionality reduction, especially given the high-dimensional nature of fMRI and sMRI data. Importantly, both the `StandardScaler` and the `RFE` selector are fitted exclusively on the training data to prevent data leakage.
*   **`train_catboost`**: This function handles the training of the `CatBoostClassifier`. It includes a hyperparameter search over specified learning rates and depths, selecting the model that yields the best validation accuracy. It also intelligently detects and utilizes GPU if available, falling back to CPU otherwise.
*   **`evaluate_model`**: Provides functionality to evaluate a trained CatBoost model on test data, returning predictions, probabilities, and accuracy.
*   **`save_model` and `load_model`**: Utility functions for serializing and deserializing trained CatBoost models using `joblib`.

### 2.4. `data_loader.py`: Data Ingestion and Preprocessing

The `data_loader.py` module is responsible for loading and initial preprocessing of the multimodal neuroimaging and phenotypic data.

*   **`get_ids` and `get_index`**: Helper functions for managing subject IDs and their indices.
*   **`load_fmri_data`**: Loads fMRI connectivity matrices, extracts the upper triangle to form feature vectors, and handles missing files or corrupted data gracefully. It also extracts associated metadata (labels, sites, genders, ages).
*   **`load_smri_data`**: Aggregates sMRI features from various FreeSurfer output files (Desikan-Killiany, ASEG, WMPARC). It calls specialized functions (`load_desikan_killiany`, `load_aseg`, `load_wmparc`) to parse and extract features from these different anatomical regions.
*   **`load_phenotypic_data`**: Loads additional phenotypic data (FIQ, NUM, PEC, RAT) and handles missing values by imputation (e.g., replacing missing FIQ with the population mean).
*   **`apply_combat`**: A general function to apply ComBat harmonization to a given data matrix, using phenotypic information as covariates. This function is called within `main.py` with careful consideration for data leakage during cross-validation.

## 3. Results Analysis (`results/save_models/CC200_sMRI/with_ComBat/`)

The results for the CatBoost model are stored under `results/save_models/CC200_sMRI/with_ComBat/`. This directory contains individual model files for each fold, a log file, and a `figures` subdirectory with various visualizations.

### 3.1. Overall Performance Metrics

From `results.csv` and `Catboost.log`, the average performance across 5-fold cross-validation is:

| Metric   | Mean Value |
| :------- | :--------- |
| Accuracy | 0.5709     |
| AUC      | 0.6095     |

These metrics indicate that the CatBoost model provides a baseline performance, with an average accuracy of approximately 57% and an AUC of nearly 61%. While better than random chance, these figures suggest room for improvement, especially when compared to more complex models like the GNN.

### 3.2. Performance per Fold (Test Accuracy)

The `results.csv` file provides the test accuracy for each fold:

| Fold | Test Accuracy |
| :--- | :------------ |
| 1    | 0.4780        |
| 2    | 0.6497        |
| 3    | 0.5402        |
| 4    | 0.6000        |
| 5    | 0.5868        |

There is considerable variability in performance across folds, with Fold 1 showing particularly low accuracy (47.8%), which is below chance level for a binary classification task. This highlights the challenges of the dataset and the potential sensitivity of the model to specific data splits or site distributions.

### 3.3. Feature Importance

The `feature_importance.png` and `modality_importance.png` visualizations provide insights into which features and modalities contributed most to the CatBoost model's predictions.

**Top 30 Feature Importances (avg across folds):**

![Top 30 Feature Importances](file:///home/ubuntu/Multimoda-braingnn/results/save_models/CC200_sMRI/with_ComBat/figures/feature_importance.png)

This plot shows that several sMRI features (e.g., `sMRI_573`, `sMRI_674`, `sMRI_580`) are consistently among the most important. This suggests that structural biomarkers play a significant role in the classification task for the CatBoost model.

**Top sMRI Features (Structural Biomarkers) and Top fMRI Features (Functional Biomarkers):**

![Modality Importance](file:///home/ubuntu/Multimoda-braingnn/results/save_models/CC200_sMRI/with_ComBat/figures/modality_importance.png)

This visualization further breaks down feature importance by modality. It confirms the dominance of sMRI features in the CatBoost model, with the top fMRI features having comparatively lower importance values. This indicates that while both modalities are used, the model relies more heavily on structural information.

## 4. Conclusion

The CatBoost implementation provides a solid, well-structured machine learning pipeline for ASD classification. It demonstrates careful consideration for data preprocessing, including ComBat harmonization and robust feature selection to prevent data leakage. The use of CatBoost, a powerful gradient boosting algorithm, offers a strong baseline for this complex task.

However, the results indicate that while CatBoost performs better than random chance, its overall accuracy and AUC are moderate, and there is notable variability across cross-validation folds. The feature importance analysis highlights the significant contribution of sMRI features to the model's decisions. This suggests that while CatBoost effectively leverages the provided features, the inherent complexity and heterogeneity of neuroimaging data for ASD classification might benefit from models capable of capturing more intricate, non-linear, and graph-based relationships, as seen with the GNN approach.
