# CatBoost fMRI/sMRI Classification

Python implementation for autism spectrum disorder (ASD) classification using combined fMRI and sMRI data with CatBoost.

## Project Structure

```
├── config.py              # Configuration and hyperparameters
├── data_loader.py         # Data loading and preprocessing
├── model_utils.py         # Model training utilities
├── visualization.py       # Results visualization
├── main.py               # Main training script
└── README.md             # This file
```

## Requirements

```bash
pip install numpy pandas scipy scikit-learn catboost neuroCombat matplotlib seaborn nibabel joblib openpyxl
```

## Configuration

Edit `config.py` to customize:

- **Paths**: Update `BASE_DIR`, `DATASET_PATH`, etc. to match your data location
- **Atlas**: Choose 'CC200' or 'AAL' for fMRI atlas
- **ComBat**: Enable/disable harmonization with `COMBAT_FMRI` and `COMBAT_SMRI`
- **Features**: Adjust feature selection parameters
- **Hyperparameters**: Modify learning rates, depths, iterations

## Usage

### Basic Training

```bash
python main.py
```

### Custom Configuration

```python
from config import Config

# Modify configuration
Config.FMRI_ATLAS = 'AAL'
Config.COMBAT_FMRI = False
Config.K_FOLD = 10

# Run training
from main import main
main()
```

## Output

The script generates:

1. **Trained Models**: Saved in `SAVE_PATH/CAT_*.m`
2. **Visualizations**:
   - `training_history.png` - Validation/test accuracy per fold
   - `roc_curve.png` - ROC curve with AUC
   - `confusion_matrix.png` - Confusion matrix
3. **Results**: `results.csv` with per-fold metrics

## Data Format

### Expected Directory Structure

```
BASE_DIR/
├── dataset/              # fMRI connectivity matrices (.mat files)
├── sMRI_dataset/
│   └── freesurfer_stats/  # FreeSurfer outputs per subject
└── phenotypic_image_quality/
    ├── subject_IDs.txt
    ├── ABIDE_label_871.mat
    ├── sites.mat
    ├── ages.mat
    ├── genders.mat
    └── ...
```

### Data Files

- **fMRI**: Connectivity matrices as `.mat` files with 'connectivity' field
- **sMRI**: FreeSurfer stats files (aparc.stats, aseg.stats, wmparc.stats)
- **Labels**: MATLAB files with phenotypic information

## Features

- ✅ **Multi-modal fusion**: Combines fMRI and sMRI features
- ✅ **ComBat harmonization**: Optional site effect correction
- ✅ **Recursive Feature Elimination**: Intelligent feature selection
- ✅ **Stratified K-Fold CV**: Maintains class balance across sites
- ✅ **Hyperparameter search**: Grid search for optimal parameters
- ✅ **Comprehensive visualization**: ROC curves, confusion matrices, training plots
- ✅ **GPU support**: CatBoost GPU acceleration

## Key Functions

### data_loader.py

- `load_fmri_data()` - Load and preprocess fMRI connectivity
- `load_smri_data()` - Load FreeSurfer sMRI features
- `apply_combat()` - Apply ComBat harmonization

### model_utils.py

- `feature_selection_fmri()` - RFE feature selection for fMRI
- `feature_selection_smri()` - RFE feature selection for sMRI
- `train_catboost()` - Train CatBoost with hyperparameter search

### visualization.py

- `plot_roc_curve()` - ROC curve visualization
- `plot_confusion_matrix()` - Confusion matrix heatmap
- `plot_training_history()` - Per-fold performance

## Example Output

```
============================================================
FINAL RESULTS
============================================================
Average Validation Accuracy: 0.7197
Average Test Accuracy:       0.7340
Overall AUC:                 0.7851
Total Test Samples:          870
```

## Customization

### Change Atlas

```python
# In config.py
Config.FMRI_ATLAS = 'AAL'  # Changes to AAL atlas
Config.IMAGE_SIZE = [116, 116]  # Update accordingly
```

### Adjust Feature Selection

```python
# In config.py
Config.NEW_FEATURES_FMRI = 3000     # Reduce fMRI features
Config.NEW_FEATURES_SMRI = 1000     # Reduce sMRI features
Config.NEW_FEATURES_COMBINE = 4000  # Reduce combined features
```

### Modify Hyperparameters

```python
# In config.py
Config.LEARNING_RATES = [0.01, 0.1]  # Fewer learning rates
Config.DEPTHS = [6, 8]               # Fewer depths
Config.ITERATIONS = 200              # More iterations
```

## Troubleshooting

### GPU Issues

If GPU not available, modify in `model_utils.py`:

```python
model = CatBoostClassifier(
    task_type='CPU',  # Change from 'GPU'
    # ... other parameters
)
```

### Memory Issues

Reduce feature numbers or batch size:

```python
Config.NEW_FEATURES_COMBINE = 3000  # Reduce features
```

### Path Errors

Ensure all paths in `config.py` point to existing directories.

## Citation

If you use this code, please cite the original ABIDE dataset and relevant papers.

## License

[Add your license here]
