# ML Expert Code Review: Multimodal Brain Classification

## Executive Summary
This is a well-structured project for autism spectrum disorder (ASD) classification using multimodal neuroimaging data. The code demonstrates good software engineering practices but has several ML-specific issues that should be addressed for production-grade quality and research reproducibility.

---

## 🔴 Critical Issues

### 1. **Data Leakage in Feature Selection**
**Severity: CRITICAL** | **File**: [main.py](main.py#L70-L95) and [model_utils.py](model_utils.py#L13-L45)

**Problem**: Feature selection is applied to the entire dataset with train indices as reference, but scaling and selection artifacts can leak validation/test information.

```python
# Current approach (PROBLEMATIC)
selector_fmri, scaler_fmri, fmri_scaled = feature_selection_fmri(
    fmri_data, labels, train_ind, config.NEW_FEATURES_FMRI
)
# Then applies to val/test sets
fmri_scaled[val_ind, :] = scaler_fmri.transform(fmri_data[val_ind, :])
```

**Issues**:
- Feature statistics (mean, std) are computed on training data, but RFE is fit on scaled training data. The selector's feature importance scores are based on training data.
- When validation/test data are scaled with training statistics, they're being transformed by a scaler fit on a subset.

**Recommendation**:
```python
def feature_selection_fmri(matrix, labels, train_ind, fnum, scaler_flag=True):
    # Create output array
    matrix_scaled = matrix.copy()
    scaler = None
    
    # Fit scaler ONLY on training data
    if scaler_flag:
        scaler = StandardScaler()
        train_data = matrix[train_ind, :]
        scaler.fit(train_data)
        # Only transform training data here
        matrix_scaled[train_ind, :] = scaler.transform(train_data)
    
    # Fit selector on training data only
    estimator = RidgeClassifier()
    selector = RFE(estimator, n_features_to_select=fnum, step=100, verbose=0)
    
    featureX = matrix_scaled[train_ind, :]
    featureY = labels[train_ind]
    selector.fit(featureX, featureY.ravel())
    
    # Return objects for caller to apply properly
    return selector, scaler, matrix_scaled
```

**Then in main loop**:
```python
# Apply scaler to val/test AFTER feature selection
def prepare_combined_features(fmri_data, smri_data, labels, train_ind, val_ind, test_ind, config):
    # ... existing code ...
    
    # Apply scalers CORRECTLY
    if scaler_fmri is not None:
        fmri_scaled[train_ind, :] = scaler_fmri.transform(fmri_data[train_ind, :])
        fmri_scaled[val_ind, :] = scaler_fmri.transform(fmri_data[val_ind, :])
        fmri_scaled[test_ind, :] = scaler_fmri.transform(fmri_data[test_ind, :])
```

---

### 2. **Fold Splitting Logic Error**
**Severity: CRITICAL** | **File**: [main.py](main.py#L24-L60)

**Problem**: The test set creation logic is confusing and potentially creates data leakage.

```python
for train, validation in sfolder.split(index_site, label):
    if group == 0:
        for j in validation:
            dist_test[str(group + k_fold)].append(index_site[j])
            dist_train[str(group + k_fold)].remove(index_site[j])  # ⚠️ DANGEROUS
    else:
        # ...
```

**Issues**:
- Modifying `dist_train` while iterating is error-prone
- Test set logic is unclear - why use `group + k_fold` indexing?
- Different stratification for different folds could introduce bias

**Recommendation**: Implement clear, standard k-fold cross-validation:

```python
def create_fold_splits(sites, labels, unique_sites, k_fold=5):
    """Create stratified K-fold splits maintaining site distribution"""
    fold_indices = {i: {'train': [], 'val': [], 'test': []} for i in range(k_fold)}
    
    for site in unique_sites:
        site_indices = np.where(np.array(sites) == site)[0]
        site_labels = labels[site_indices]
        
        # Stratified k-fold for this site
        skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=0)
        
        for fold_idx, (train_val, test) in enumerate(skf.split(site_indices, site_labels)):
            train_val_indices = site_indices[train_val]
            test_indices = site_indices[test]
            
            # Further split train_val into train and validation (80/20)
            tv_labels = labels[train_val_indices]
            train_size = int(0.8 * len(train_val_indices))
            train, val = train_val_indices[:train_size], train_val_indices[train_size:]
            
            fold_indices[fold_idx]['train'].extend(train)
            fold_indices[fold_idx]['val'].extend(val)
            fold_indices[fold_idx]['test'].extend(test_indices)
    
    return fold_indices
```

---

### 3. **Hyperparameter Tuning on Validation Set (Not Separate Test Set)**
**Severity: HIGH** | **File**: [model_utils.py](model_utils.py#L67-L100)

**Problem**: Grid search uses validation accuracy to select hyperparameters, but then reports test accuracy. This is correct in principle, but there's no independent hold-out test set for final evaluation.

**Current Flow**:
- Train → Validation (hyperparameter selection) → Test (performance reporting)
- ❌ Test set used for reporting final metrics is the same set used implicitly during training

**Recommendation**: Implement proper nested cross-validation or use a separate hold-out test set:

```python
# Option 1: Nested CV
from sklearn.model_selection import cross_validate

inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

def train_catboost_cv(train_data, train_labels, learning_rates, depths, config):
    """Train with inner CV for hyperparameter selection"""
    best_params = {'lr': None, 'depth': None, 'score': 0}
    
    for lr in learning_rates:
        for depth in depths:
            model = CatBoostClassifier(
                iterations=config.ITERATIONS,
                learning_rate=lr,
                depth=depth,
                verbose=0,
                random_state=config.RANDOM_SEED,
                task_type='GPU',
                devices='0'
            )
            
            scores = cross_val_score(model, train_data, train_labels, 
                                    cv=inner_cv, scoring='accuracy')
            mean_score = scores.mean()
            
            if mean_score > best_params['score']:
                best_params = {'lr': lr, 'depth': depth, 'score': mean_score}
    
    return best_params
```

---

## 🟡 High Priority Issues

### 4. **Phenotypic Feature Repetition (Unclear Justification)**
**Severity: HIGH** | **File**: [main.py](main.py#L175-L179)

```python
for _ in range(3):
    smri_data = np.concatenate((smri_data, ages, genders, fiq, num, pec, rat), axis=1)
```

**Problem**:
- Why repeat 3 times? This inflates feature dimensions unnaturally
- No comment explaining the rationale
- Creates redundancy that could bias feature selection

**Recommendation**:
```python
# Either concatenate once with clear justification
smri_data = np.concatenate((smri_data, ages, genders, fiq, num, pec, rat), axis=1)

# OR provide a comment explaining multi-scale approach
# Note: Phenotypic features repeated 3x to match multi-scale fMRI representations
# This approach mirrors [Citation to paper/method]
```

---

### 5. **Missing Class Balance Analysis**
**Severity: HIGH** | **File**: [main.py](main.py#L172-L180)

**Problem**: No check for class imbalance in train/val/test splits.

**Recommendation**:
```python
def create_fold_splits(sites, labels, unique_sites, k_fold=5):
    # ... existing code ...
    
    # After creating splits, add validation
    for fold_idx in fold_indices:
        train_labels = labels[fold_indices[fold_idx]['train']]
        val_labels = labels[fold_indices[fold_idx]['val']]
        test_labels = labels[fold_indices[fold_idx]['test']]
        
        print(f"\nFold {fold_idx + 1}:")
        print(f"  Train class distribution: {np.bincount(train_labels.astype(int))}")
        print(f"  Val class distribution:   {np.bincount(val_labels.astype(int))}")
        print(f"  Test class distribution:  {np.bincount(test_labels.astype(int))}")
        
        # Check for extreme imbalance
        train_pos_ratio = np.sum(train_labels) / len(train_labels)
        if train_pos_ratio < 0.3 or train_pos_ratio > 0.7:
            warnings.warn(f"Fold {fold_idx + 1}: Severe class imbalance detected!")
```

---

### 6. **No Feature Scaling for CatBoost (Unnecessary but Valid)**
**Severity: MEDIUM** | **File**: [model_utils.py](model_utils.py#L13-L45)

**Problem**: CatBoost is a tree-based model and doesn't require feature scaling, but the code applies StandardScaler. While not harmful, it's wasteful.

**Recommendation**:
```python
def feature_selection_fmri(matrix, labels, train_ind, fnum, scaler_flag=False):
    """
    Feature selection for fMRI data using RFE
    
    Note: CatBoost doesn't require scaling, but we keep it for compatibility
    with potential future models (e.g., neural networks)
    """
    # Consider removing scaling entirely for CatBoost-specific pipeline
```

---

### 7. **Hard-coded GPU Device**
**Severity: MEDIUM** | **File**: [model_utils.py](model_utils.py#L85-L92)

```python
model = CatBoostClassifier(
    ...
    task_type='GPU',
    devices='0'  # Hard-coded to GPU 0
)
```

**Problem**:
- Fails if GPU 0 not available or user has single GPU
- Should fallback to CPU gracefully

**Recommendation**:
```python
import torch

def train_catboost(train_data, train_labels, val_data, val_labels, 
                   learning_rates, depths, config, verbose=0):
    """Train CatBoost with automatic device selection"""
    
    # Auto-detect GPU availability
    has_gpu = torch.cuda.is_available()
    task_type = 'GPU' if has_gpu else 'CPU'
    devices = '0' if has_gpu else None
    
    for lr in learning_rates:
        for depth in depths:
            try:
                model = CatBoostClassifier(
                    iterations=config.ITERATIONS,
                    learning_rate=lr,
                    depth=depth,
                    verbose=verbose,
                    random_state=config.RANDOM_SEED,
                    task_type=task_type,
                    devices=devices if has_gpu else None
                )
                # ... rest of code ...
```

---

## 🟠 Medium Priority Issues

### 8. **RFE Step Size Inconsistency**
**Severity: MEDIUM** | **File**: [model_utils.py](model_utils.py#L40-L41)

```python
# fMRI: step=100
selector_fmri = RFE(estimator, n_features_to_select=fnum, step=100, verbose=0)

# sMRI: step=10
selector_smri = RFE(estimator, n_features_to_select=fnum, step=10, verbose=0)
```

**Problem**: No explanation for why different step sizes. RFE step size significantly affects performance.

**Recommendation**:
```python
# Document the choice
FMRI_RFE_STEP = 100  # Aggressive elimination for high-dimensional fMRI (200x200 matrix)
SMRI_RFE_STEP = 10   # Conservative elimination for sMRI features

# Or make it configurable in Config class
class Config:
    RFE_STEP_FMRI = 100
    RFE_STEP_SMRI = 10
```

---

### 9. **Missing Error Handling for Missing Data**
**Severity: MEDIUM** | **File**: [data_loader.py](data_loader.py#L30-L90)

**Problem**: If a subject file is missing or corrupted, the entire pipeline fails without meaningful error message.

```python
# Current code - no try/except
for i in range(config.NUM_SAMPLES):
    subject_name = subject_IDs[i]
    if subject_name in config.USELESS_SAMPLES:
        continue
    
    image_name = f'{subject_name}.mat'
    image = scio.loadmat(os.path.join(config.DATASET_PATH, image_name))  # Can fail silently
```

**Recommendation**:
```python
def load_fmri_data(config):
    # ... existing code ...
    
    missing_subjects = []
    for i in range(config.NUM_SAMPLES):
        subject_name = subject_IDs[i]
        if subject_name in config.USELESS_SAMPLES:
            continue
        
        try:
            image_name = f'{subject_name}.mat'
            image_path = os.path.join(config.DATASET_PATH, image_name)
            
            if not os.path.exists(image_path):
                missing_subjects.append(subject_name)
                continue
            
            image = scio.loadmat(image_path)
            if 'connectivity' not in image:
                raise ValueError(f"'connectivity' field not found in {image_name}")
            
            # ... process data ...
        
        except Exception as e:
            print(f"Warning: Failed to load {subject_name}: {str(e)}")
            missing_subjects.append(subject_name)
            continue
    
    if missing_subjects:
        print(f"\nWarning: {len(missing_subjects)} subjects failed to load:")
        for s in missing_subjects:
            print(f"  - {s}")
```

---

### 10. **No Stratification by Gender/Age in Cross-Validation**
**Severity: MEDIUM** | **File**: [main.py](main.py#L24-L60)

**Problem**: K-fold splits are stratified by site and label, but not by gender/age. This could lead to demographic imbalance across folds.

**Recommendation**:
```python
# Add demographic stratification
def create_fold_splits(sites, labels, unique_sites, genders, ages, k_fold=5):
    """Create stratified K-fold splits with demographic balance"""
    
    for site in unique_sites:
        # Create a composite stratification target
        site_mask = np.array(sites) == site
        composite_strat = (
            labels[site_mask].astype(str) + '_' +
            genders[site_mask].flatten().astype(str) + '_' +
            np.digitize(ages[site_mask].flatten(), bins=[20, 30, 40, 50]).astype(str)
        )
        
        skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=0)
        for train, test in skf.split(site_indices, composite_strat):
            # ... rest of code ...
```

---

## 🔵 Low Priority Issues (Code Quality & Best Practices)

### 11. **Unused Import and Dead Code**
**Severity: LOW** | **File**: [visualization.py](visualization.py#L1-L10)

The function `plot_ensemble_comparison` is defined but never called.

**Recommendation**: Remove or implement if needed.

---

### 12. **Inconsistent Naming Conventions**
**Severity: LOW** | **File**: Multiple files

- `dist_train`, `dist_val`, `dist_test` (abbreviations)
- `fmri_data`, `smri_data`, `fMRI_images` (inconsistent)
- `featureX`, `featureY` (bad naming)

**Recommendation**:
```python
# Better naming
train_indices, val_indices, test_indices = create_fold_splits(...)

# In feature selection
def feature_selection_fmri(feature_matrix, labels, train_indices, n_features, use_scaler=True):
    X_train = feature_matrix[train_indices, :]
    y_train = labels[train_indices]
```

---

### 13. **Magic Numbers in Configuration**
**Severity: LOW** | **File**: [config.py](config.py)

```python
# Why these specific numbers?
NEW_FEATURES_FMRI = 5000
NEW_FEATURES_SMRI = 1435
NEW_FEATURES_COMBINE = 6000
```

**Recommendation**:
```python
class Config:
    # Feature selection parameters with justification
    # fMRI: CC200 atlas = 200x200 connectivity matrix = 19,900 features (upper triangle)
    # Select ~25% to reduce dimensionality while preserving discriminative information
    NEW_FEATURES_FMRI = 5000
    
    # sMRI: 68 Desikan + 45 ASEG + 70 WMPARC = 183 ROIs × 7 features = 1281 features
    # Add phenotypic features (repeated 3x) ≈ 1435 total
    NEW_FEATURES_SMRI = 1435
    
    # Combined: ~30% reduction on concatenated features
    NEW_FEATURES_COMBINE = 6000
```

---

### 14. **No Logging Infrastructure**
**Severity: LOW** | **File**: All files

**Problem**: Uses `print()` statements scattered throughout. No logging levels, timestamps, or file logging.

**Recommendation**:
```python
import logging

# In config.py
def setup_logger(log_dir):
    logger = logging.getLogger('MultimodalBrain')
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# Usage
logger = setup_logger(config.SAVE_PATH)
logger.info(f"Loaded {len(subject_ids)} subjects")
logger.warning("Class imbalance detected in fold 2")
logger.error("Failed to load subject XXX")
```

---

### 15. **No Reproducibility Seed Management**
**Severity: LOW** | **File**: [main.py](main.py#L1-L20)

**Problem**: Only numpy random seed set, but not Python's random module or PyTorch (if used in future).

**Recommendation**:
```python
import random
import numpy as np
import torch

def set_seed(seed=0):
    """Set seed for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# In main()
config = Config()
set_seed(config.RANDOM_SEED)
```

---

## ✅ Positive Aspects

1. **Good modular structure**: Separate files for config, data loading, model utils, visualization
2. **Comprehensive cross-validation**: Stratified K-fold with site awareness
3. **Multiple modalities**: Intelligent fusion of fMRI and sMRI data
4. **Harmonization**: Uses ComBat for site effect correction
5. **Hyperparameter search**: Grid search for learning rate and depth
6. **Visualization**: Comprehensive plots (ROC, confusion matrix, training history)
7. **ComBat integration**: Proper handling of batch effects with covariates

---

## 📊 Summary Table

| Issue | Severity | Category | Impact |
|-------|----------|----------|--------|
| Data leakage in feature selection | 🔴 CRITICAL | ML | Inflated metrics, unreliable results |
| Fold splitting logic | 🔴 CRITICAL | ML | Potential data contamination |
| Hyperparameter tuning setup | 🟡 HIGH | ML | Optimistic bias in test metrics |
| Feature repetition unexplained | 🟡 HIGH | ML | Unclear methodology |
| Missing class balance check | 🟡 HIGH | ML | Hidden imbalance issues |
| Hard-coded GPU device | 🟡 HIGH | Code | Runtime failures |
| RFE step inconsistency | 🟠 MEDIUM | ML | Unexplained behavior |
| Missing error handling | 🟠 MEDIUM | Code | Silent failures |
| No demographic stratification | 🟠 MEDIUM | ML | Potential bias |
| Naming conventions | 🔵 LOW | Quality | Readability |
| Magic numbers | 🔵 LOW | Quality | Maintainability |
| No logging | 🔵 LOW | Quality | Debugging difficulty |

---

## 🎯 Priority Action Items

1. **Fix data leakage** in feature selection pipeline
2. **Clarify fold splitting logic** with standard CV implementation
3. **Add error handling** for missing/corrupted data
4. **Document design choices** (repetition, step sizes, thresholds)
5. **Add class balance validation** across all splits
6. **Implement GPU fallback** to CPU
7. **Add reproducibility seeds** across all libraries
8. **Improve variable naming** for clarity
9. **Add logging infrastructure** for better debugging
10. **Consider nested cross-validation** for more robust hyperparameter selection

---

## 📚 Recommended Reading

- [A Disciplined Approach to Neural Network Train/Test Splits](https://openreview.net/forum?id=_wsB7-JNKZA)
- [Nested Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html#nested-cross-validation)
- [ComBat for Harmonization](https://pubmed.ncbi.nlm.nih.gov/29409560/)
- [Feature Selection Best Practices](https://www.datacamp.com/blog/feature-selection-for-machine-learning-in-python)

---

**Review Date**: January 5, 2026
**Reviewer**: ML Expert Code Analysis
**Overall Grade**: B+ (Good structure, critical ML issues to address)
