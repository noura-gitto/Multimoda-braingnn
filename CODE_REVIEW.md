# Code Review: BrainGNN-Multimodal

## Executive Summary

This project implements a sophisticated multimodal deep learning pipeline for autism spectrum disorder (ASD) classification using fMRI connectivity matrices and sMRI anatomical features from the ABIDE dataset. The implementation demonstrates good software engineering practices with comprehensive architecture, but shows modest classification performance (~64% accuracy) with room for improvement.

---

## 📊 Results Analysis

### Overall Performance Metrics
```
Average Test Accuracy:  63.85% ± 3.60%
Average Test AUC:       66.51% ± 3.10%
Average Test F1-Score:  63.46% ± 3.30%
```

### Per-Fold Results
- **Fold 1:** Accuracy=67.76%, AUC=66.34%, F1=65.92%
- **Fold 2-5:** Similar performance with modest variance (std ≈ 3%)

### Key Observations
✅ **Consistent Performance:** Low variance across folds (3-4%) indicates stable cross-validation  
⚠️ **Moderate Baseline:** 64% accuracy is only marginally better than balanced random (~50%)  
⚠️ **High Variance in Predictions:** Model confidence varies significantly; needs calibration

---

## 🏗️ Architecture Review

### 1. **fMRI Graph Neural Network Branch** (`braingnn_multimodal.py`)

**Strengths:**
- Appropriate use of Graph Convolutional Networks (GCN) for connectivity matrices
- Graph Attention Layer (GAT) captures node importance dynamically
- Top-K pooling reduces dimensionality intelligently
- Self-attention mechanism captures global graph dependencies

**Weaknesses:**
```python
# Edge threshold (Line 250)
self.edge_threshold = 0.2  # Quite aggressive
```
- **Aggressive edge thresholding (0.2):** Removes valuable weak connections. Consider lowering to 0.1-0.15
- **Graph pooling ratio (0.5):** Discards 50% of nodes—may lose important spatial information
- **BatchNorm applied incorrectly:** Currently transposes (N,M,D)→(N,D,M), disrupting channel semantics
  
**Code Quality Issue:**
```python
# braingnn_multimodal.py, Lines 287-291
x = x.transpose(1, 2)
x = self.bn1(x)
x = x.transpose(1, 2)
```
This works but is inefficient. Use `nn.LayerNorm` instead for graph data.

### 2. **sMRI Deep Neural Network Branch**

**Strengths:**
- Residual blocks enable training of deeper networks
- Multi-head self-attention models feature interactions
- Channel-wise attention captures feature importance
- Handles variable-length sMRI features gracefully

**Weaknesses:**
- Hidden dimension (512) is quite large—may lead to overfitting with 871 samples
- Two residual blocks may be insufficient depth for 2072 feature dimension
- No explicit dimensionality reduction before fully connected layers

### 3. **Multimodal Fusion**

**Strengths:**
- Cross-modal attention is bidirectional (fMRI→sMRI and vice versa)
- Bilinear pooling captures second-order interactions
- Reasonable feature concatenation strategy

**Architecture Gap:**
```python
# MultimodalFusion forward (lines 545-575)
fmri_seq = fmri_features.unsqueeze(0)  # Adds sequence dim
# Then cross-attention between single samples
```
This treats each sample independently. Consider:
- Learning explicit fusion weights
- Gating mechanisms to control modality contributions
- Early fusion of raw features (not just embeddings)

### 4. **Multi-task Learning**

**Configuration (train_braingnn.py, line 990):**
```python
'lambda_cls': 1.0,      # Classification
'lambda_site': 0.05,    # Site adaptation (weak)
'lambda_age': 0.01,     # Age deconfounding (very weak)
'lambda_reg': 0.0001    # L2 regularization (minimal)
```

**Assessment:**
- Site adaptation weight (0.05) is very weak given multi-site dataset challenges
- Age deconfounding is underweighted (0.01)
- FIQ was removed but was potentially useful—consider re-inclusion with proper handling

### 5. **Data Preprocessing**

**fMRI Processing:**
```python
# train_braingnn.py, lines 681-688
fmri_clipped = np.clip(fmri_data, -0.999, 0.999)
fmri_data = np.arctanh(fmri_clipped)  # Fisher z-transform
fmri_mean = fmri_data.mean(axis=(1,2), keepdims=True)
fmri_std = fmri_data.std(axis=(1,2), keepdims=True) + 1e-8
fmri_data = (fmri_data - fmri_mean) / fmri_std
```

✅ **Good:** Fisher z-transform + per-subject normalization  
⚠️ **Consider:** Graph thresholding/sparsification **before** normalization

**sMRI Processing:**
```python
# train_braingnn.py, lines 690-697
smri_mean = np.nanmean(smri_data, axis=0, keepdims=True)
smri_std = np.nanstd(smri_data, axis=0, keepdims=True) + 1e-8
smri_data = (smri_data - smri_mean) / smri_std
smri_data = np.nan_to_num(smri_data)
```

✅ **Good:** Handles NaN values  
⚠️ **Concern:** sMRI extraction is basic—parsing arbitrary text files may miss structure

---

## 🔴 Critical Issues

### 1. **Low Classification Performance**
- 64% accuracy is close to random baseline
- Model likely underfitting the data
- Possible causes:
  - Feature quality/dimensionality mismatch
  - Inadequate regularization
  - Model architecture not complex enough

**Recommendations:**
```python
# Increase model capacity
'hidden_dim': 512,      # Was: 256
'smri_hidden': 1024,    # Was: 512
'num_gnn_layers': 4,    # Add one more GNN layer
```

### 2. **Data Imbalance Not Addressed**
- No mention of class weights or SMOTE
- ASD:TD ratio in ABIDE is typically 1:1 but may be skewed
- FocalLoss is implemented but `alpha` parameter is hardcoded to 1.0

**Fix:**
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, label_smoothing=0.1):
        # Compute alpha from class weights if None
        if alpha is None:
            # Should be computed from data
            self.alpha = [1.0, 1.5]  # Adjust based on class distribution
```

### 3. **Site Awareness Incomplete**
- Site-aware cross-validation is implemented ✅
- But domain adaptation weight is too low (0.05)
- No batch normalization per-site or site-specific adaptation layers

**Suggestion:**
```python
# Increase site adaptation
'lambda_site': 0.2,  # Was: 0.05
# Add domain adaptation layer
class DomainAdaptationLayer(nn.Module):
    def __init__(self, hidden_dim, num_sites):
        self.domain_classifier = ...
        self.domain_discriminator = ...  # Adversarial
```

### 4. **No Hyperparameter Tuning Evidence**
- Learning rate fixed at 5e-4
- No ablation studies shown
- Limited justification for architectural choices

---

## ⚠️ Code Quality Issues

### 1. **Duplicate Configuration Parameters**
```python
# train_braingnn.py, lines 989-1003
'lambda_cls': 1.0,      # Line 994
'lambda_site': 0.1,     # Line 997
'lambda_site': 0.05,    # Line 1001 (DUPLICATE - overwrites!)
```
**Fix:** Remove duplicate, keep one definition.

### 2. **Magic Numbers**
- Edge threshold: 0.2 (no justification)
- Pooling ratio: 0.5 (why?)
- Warmup epochs: 10 (why 10 and not adaptive?)
- Dropout: 0.3 throughout (not tuned per layer)

### 3. **Error Handling**
Good logging, but some critical errors are swallowed:
```python
# train_braingnn.py, line 642
try:
    mat_file = os.path.join(root_path, f"{subject_id}.mat")
    data = scio.loadmat(mat_file)
    connectivity = data['connectivity']
    fmri_data[i] = connectivity
except Exception as e:
    logger.warning(f"Error loading {subject_id}: {e}")
    fmri_data[i] = np.zeros((num_nodes, num_nodes))  # Silent failure!
```
**Impact:** Missing data filled with zeros—biases model toward TDs. Consider imputation.

### 4. **Inefficient Attention Mechanism**
```python
# braingnn_multimodal.py, lines 159-173
def _prepare_attentional_mechanism_input(self, Wh: torch.Tensor):
    batch_size, num_nodes, out_features = Wh.size()
    Wh_repeated_in_chunks = Wh.repeat_interleave(num_nodes, dim=1)
    Wh_repeated_alternating = Wh.repeat(1, num_nodes, 1)
    # Creates (batch, num_nodes^2, 2*features) dense tensor
```
**Issue:** O(n²) memory for n=200 nodes. Scales poorly.

---

## 📈 Performance Improvement Suggestions

### High Priority (Expected +2-5% accuracy)
1. **Increase site adaptation weight to 0.2-0.3**
   - Multi-site harmonization is critical for ABIDE data
   - Consider domain adversarial training

2. **Use weighted sampling or class weights**
   - Even if balanced, confidence calibration needed
   - Adjust FocalLoss alpha parameter

3. **Improve sMRI feature extraction**
   - Current approach (regex on text files) is brittle
   - Extract structured features: gray matter volume, cortical thickness, etc.
   - May need to use freesurfer cortical parcellation

4. **Add input validation & imputation**
   - Don't silently replace missing fMRI with zeros
   - Use matrix completion or KNN imputation

### Medium Priority (Expected +1-3% accuracy)
5. **Reduce edge threshold (0.2 → 0.1)**
   - Preserves weak but meaningful connections
   - Re-test performance after tuning

6. **Use LayerNorm instead of BatchNorm for graph data**
   - More stable for variable-sized graphs
   - Better theoretical justification

7. **Add skip connections between branches**
   - Current architecture has no direct fMRI→classification path
   - Implement early classification head alongside fusion

8. **Implement proper hyperparameter search**
   - Random search or Bayesian optimization
   - Focus on: hidden_dim, pooling_ratio, learning_rate, lambda_site

### Low Priority (Polish)
9. Remove duplicate configuration parameters
10. Document magic numbers with justification
11. Add unit tests for data loading pipeline
12. Cache preprocessed features to speed up training

---

## 📋 Code Organization

### Positives ✅
- Clear separation of concerns (data, model, training)
- Comprehensive logging throughout
- Good documentation with docstrings
- Visualization pipeline is thorough
- Reproducibility: fixed seeds and deterministic settings

### Areas for Improvement 📝
- Config validation (no checks for `lambda_*` values)
- No unit tests
- Constants should be in separate config file
- Model factory is minimal—consider more flexibility

---

## 🧪 Experimental Recommendations

### 1. Ablation Study Template
```
Experiment 1: Full model (baseline)       → 64% accuracy
Experiment 2: - Site adaptation           → ? (expect worse)
Experiment 3: - sMRI branch              → ? (expect worse if sMRI helps)
Experiment 4: - Fusion attention         → ? (test fusion impact)
Experiment 5: - GAT (use pure GCN)       → ? (test GAT necessity)
```

### 2. Hyperparameter Tuning
```python
# Systematic search
for edge_thresh in [0.05, 0.1, 0.15, 0.2, 0.25]:
    for pool_ratio in [0.3, 0.5, 0.7]:
        for lambda_site in [0.05, 0.1, 0.2, 0.3]:
            # Train and record
```

### 3. Feature Importance Analysis
```python
# Add after training:
# - Gradient-based feature importance (saliency maps)
# - Attention weight visualization per layer
# - SHAP values for interpretability
```

---

## 🎯 Summary Table

| Aspect | Rating | Notes |
|--------|--------|-------|
| Architecture Design | 🟡 Good | Sound multimodal approach, minor issues |
| Implementation Quality | 🟢 Good | Clean code, minor bugs (duplicate params) |
| Results | 🔴 Poor | 64% accuracy, needs improvement |
| Data Handling | 🟡 Fair | Missing data silently ignored |
| Documentation | 🟢 Good | Clear docstrings and logging |
| Reproducibility | 🟢 Good | Fixed seeds and logged config |

---

## 🚀 Next Steps

1. **Immediate:** Fix duplicate `lambda_site` parameter
2. **Week 1:** Run ablation study to understand bottlenecks
3. **Week 2:** Implement domain adaptation improvements
4. **Week 3:** Experiment with sMRI feature extraction
5. **Week 4:** Systematic hyperparameter tuning

---

## Questions for Authors

1. Why was FIQ removed? Was performance worse with it?
2. What is the class distribution in the test set?
3. Have you tried different graph construction methods (sparse, k-NN)?
4. Why pooling ratio of 0.5? Did you test others?
5. What's the memory/time cost of the attention mechanism?

---

*Review Date: January 12, 2026*  
*Reviewed Code: train_braingnn.py, braingnn_multimodal.py*  
*Results Analyzed: results_GNN/ (5-fold cross-validation)*
