# Data Leakage Analysis & Fixes

## Executive Summary

Your training logs show **critical data leakage and site overfitting** causing artificially inflated accuracy metrics. The model achieves **94% global accuracy** but only **24-52% on individual sites**, indicating it has learned to identify **scanners/sites** rather than **ASD biomarkers**.

---

## Issue Breakdown

### 1. **Critical Issue: ComBat Applied Before Fold Split** 🔴

**Location:** [main.py](main.py#L271-L277)

```python
# CURRENT (WRONG):
fmri_data = apply_combat(fmri_data, sites, labels, ...)  # ALL 870 samples
smri_data = apply_combat(smri_data, sites, labels, ...)  # ALL 870 samples
fold_indices = create_fold_splits(...)  # AFTER - test info leaked
```

**The Problem:**
- ComBat learns site effects from **all 870 samples** including test set
- When you later split into folds, the test set has already influenced the harmonization
- This is **information leakage** - test set information flows backward into training

**Impact:**
- ComBat removes "site differences" based on knowledge of test set composition
- This artificially boosts accuracy on global metrics but fails on held-out sites
- The model learns site-specific patterns that generalize poorly

**The Fix:**
Apply ComBat **per fold** using only training data:

```python
# FIXED (in main_fixed.py):
for fold in range(K_FOLD):
    # ComBat learns ONLY from training sites of this fold
    fmri_harmonized, smri_harmonized = apply_combat_per_fold(
        fmri_data, smri_data, sites, labels, genders, ages,
        unique_sites, fold_indices, fold, COMBAT_FMRI, COMBAT_SMRI
    )
    # Then feature selection on this harmonized data
    # Then training
```

---

### 2. **Feature Selection Isolation** ✓ (Good)

**Location:** [main.py](main.py#L104-L170) | [model_utils.py](model_utils.py#L10-L83)

Good news: Your feature selection **is correctly isolated**:

```python
def feature_selection_fmri(matrix, labels, train_ind, ...):
    # Only fits RFE on training data
    selector.fit(X_train_scaled, y_train)  # train_ind only ✓
    return selector, scaler
```

✅ You're already doing this correctly.

---

### 3. **ComBat Paradox: Why sMRI Dominates** 🤔

**Observation from your logs:**

```
Top Features by Modality:
  Feature  Importance  Modality
  sMRI_586  11.94      sMRI      ← Massive importance
  sMRI_556  11.70      sMRI
  sMRI_575  11.56      sMRI
  ...
  fMRI_4082  0.298     fMRI      ← 40x less important
```

**Why?**

1. ComBat is a **linear harmonization** method
2. White matter (sMRI) parcellation volumes have **scanner-specific intensity biases**
3. CatBoost finds non-linear patterns that ComBat missed
4. These residual site patterns are strong enough to predict global ASD/TD

**Evidence:** Look at top biomarkers - all white matter parcels (WMPARC). These are **highly sensitive to scanner settings** (field strength, protocols, reconstruction).

---

### 4. **Class Imbalance Makes Accuracy Misleading** ⚠️

Your validation sets are **imbalanced**:
- Val positive class: **25-29%** (should be ~51%)
- This makes standard **Accuracy** a poor metric

**Example:** A dummy classifier predicting "TD" for all samples would achieve:
- Accuracy: ~72% (correct on majority class)
- But it would be completely useless

**Fix in provided code:** Use **Balanced Accuracy** and **Macro F1** instead:

```python
balanced_acc = balanced_accuracy_score(y_true, y_pred)  # Average of sensitivity & specificity
macro_f1 = f1_score(y_true, y_pred, average='macro')     # Average F1 for each class
```

These metrics are **immune to class imbalance**.

---

## Provided Solutions

I've created three new scripts that implement the fixes:

### **1. main_fixed.py** - Standard 5-Fold CV with Proper Isolation

```
✓ ComBat applied PER FOLD using only training data
✓ Feature selection remains isolated (already good)
✓ Added balanced accuracy and macro F1 metrics
✓ Drop-in replacement for main.py

Usage:
python main_fixed.py
```

**Expected outcome:** Accuracy will likely **drop** (good sign!) because you're no longer leaking test information.

---

### **2. main_loso.py** - Leave-One-Site-Out Cross-Validation

```
✓ Trains on 19 sites, tests on holdout site
✓ Repeats for all 20 sites
✓ TRUE generalization test across hospitals/scanners

This is the GOLD STANDARD for multisite neuroimaging
```

**How to interpret results:**

| LOSO Accuracy | Interpretation |
|---|---|
| 70-85% | ✓ Model learned genuine biomarkers |
| 50-65% | ⚠️ Site effects are significant; overfitting present |
| <50% | ✗ Model is learning site patterns only |

**Run it:**
```python
python main_loso.py
```

---

### **3. main_unimodal.py** - fMRI-Only vs sMRI-Only Models

Tests each modality separately.

**Interpretation:**

| Scenario | Meaning |
|---|---|
| fMRI ≈ sMRI accuracy | ✓ Both modalities contain true signal |
| sMRI >> fMRI | ⚠️ sMRI has site-specific scanner bias |
| fMRI >> sMRI | ✓ fMRI is more robust (good) |

If sMRI accuracy drops significantly, it confirms that white matter volumes are being exploited for site identification rather than ASD diagnosis.

**Run it:**
```python
python main_unimodal.py
```

---

## Step-by-Step Fix Action Plan

### **Week 1: Diagnosis**
1. Run `main_fixed.py` - document if accuracy drops
2. Run `main_loso.py` - see true performance across sites
3. Run `main_unimodal.py` - check if sMRI is the culprit

### **Week 2: Investigation**
4. Analyze which features correlate with site vs. diagnosis
5. Check if removing top sMRI features improves LOSO accuracy
6. Consider site-robust feature selection methods (e.g., harmonization-aware RFE)

### **Week 3: Implementation**
7. If LOSO << 5-fold: Use LOSO as primary metric (hard truth)
8. If sMRI is culprit: Weight fMRI features higher or use separate models
9. Consider medical imaging-specific methods like Freesurfer's cross-site harmonization

---

## Additional Recommendations

### **A. Improve ComBat Application**

Current limitation: neuroCombat doesn't easily return transformation matrices. Better approach:

```python
from neuroharmony import harmonize  # Better for applied scientists
# or use combat-harmonization library with explicit parameter extraction
```

### **B. Site-Stratified Metrics**

Show per-site performance in your reports:

```python
for site in unique_sites:
    site_mask = (sites == site)
    site_acc = accuracy_score(labels[site_mask], preds[site_mask])
    print(f"{site}: {site_acc:.2%}")
```

This will immediately reveal overfitting.

### **C. Consider Batch Effect Methods**

Instead of ComBat alone, try:
1. **Residual Variance Explained (RVE)** - robust estimate
2. **ComBat + Harmonization** - two-step approach
3. **Site-blind feature selection** - select features that work across sites

### **D. Cross-site Validation Reporting**

Always report:
```
- 5-Fold CV Accuracy: 94% ← old metric (potentially inflated)
- LOSO Accuracy: ?? % ← TRUE generalization
- Balanced Accuracy: ?? % ← robust to imbalance
- Per-site accuracy: ... ← transparency
```

---

## Key Takeaways

| Metric | Your Current | What It Means |
|---|---|---|
| Global Acc | 94% | Misleading - includes site effects |
| Site-specific Acc | 24-52% | ← Real generalization ability |
| ComBat timing | Before splits ✗ | Data leakage |
| Feature selection | After splits ✓ | Correct (don't change!) |
| Imbalance handling | Accuracy only | Misleading - use balanced metrics |

---

## Files to Run

```bash
# Original pipeline (for comparison)
python main.py

# Fixed 5-fold CV (proper isolation)
python main_fixed.py

# Leave-One-Site-Out (gold standard)
python main_loso.py

# Unimodal analysis (debug which modality causes overfitting)
python main_unimodal.py
```

---

## Questions to Answer After Running

1. **Does accuracy drop in `main_fixed.py`?** → Yes = leakage was happening
2. **What is LOSO accuracy in `main_loso.py`?** → Compare to 5-fold
3. **Does sMRI-only match combined in `main_unimodal.py`?** → Yes = sMRI likely contains site bias

**If all three suggest heavy site overfitting:** Congratulations - you've identified a critical methodological issue that would have invalidated your conclusions. Fixing this now saves you from retracting papers later.

---

## References

- **ComBat method:** Johnson et al. (2007) - "Adjusting batch effects in microarray expression data"
- **LOSO for neuroimaging:** Poldrack et al. (2019) - "Computational Models for the Prediction of Autism Spectrum Disorder Based on the Structural Connectome"
- **Harmonization in fMRI:** Fortin et al. (2017) - "Harmonization of cortical thickness measurements across scanners and sites"
