# Neuroscience-Focused Visualization Guide

## Overview

The improved visualization pipeline now includes **8 key figures** designed specifically for neuroscience researchers to demonstrate:

1. **Clinical Value** - How well the model balances sensitivity/specificity
2. **Neuroscience Value** - Which brain regions are actual biomarkers for autism
3. **Generalization** - No overfitting to specific hospitals/scanners

---

## Figures Explanation

### 1. **Site Performance Chart** (`site_performance.png`)
**Purpose**: Proves the model isn't overfitting to a specific hospital or scanner

**What it shows**:
- **Accuracy per Site**: Bar chart showing model performance across all hospitals/scanning sites
- **Sensitivity & Specificity per Site**: Clinical balance (ability to catch ASD vs correctly identify typical development)
- **AUC per Site**: Discrimination ability at each site
- **Sample Distribution**: Ensures comparison is fair across different-sized sites

**Interpretation**:
- ✅ **GOOD**: All sites have similar accuracy (no site >10% better than others)
- ❌ **BAD**: One site has 20% higher accuracy → model may have learned site-specific artifact
- **Clinical Value**: Shows the model generalizes across real-world scanning environments

**Example Output**:
```
Site    N   Accuracy  Sensitivity  Specificity  AUC
Site1   50    0.82       0.85         0.79      0.88
Site2   45    0.80       0.78         0.82      0.86
Site3   48    0.81       0.84         0.77      0.87
Mean          0.81       0.82         0.79      0.87
```

---

### 2. **Clinical Metrics Dashboard** (`clinical_metrics.png`)
**Purpose**: Shows clinical utility—balance between missing cases (false negatives) and false alarms (false positives)

**What it shows**:
- **Confusion Matrix**: % of true cases in each cell
- **ROC Trade-off**: Sensitivity vs Specificity across thresholds
- **Clinical Decision Metrics**: 
  - **Sensitivity**: Can you catch ASD cases? (minimize false negatives)
  - **Specificity**: Can you avoid false alarms? (minimize false positives)
  - **PPV/NPV**: If model says ASD/TD, is it right?
- **Summary Table**: Color-coded metrics (green=good, red=poor)

**Clinical Interpretation**:
- **Sensitivity 90%**: Out of 100 ASD kids, you catch 90 (miss 10) ← Type II error
- **Specificity 85%**: Out of 100 typical kids, you correctly identify 85 (false alarm for 15) ← Type I error
- **PPV 92%**: If model says ASD, it's correct 92% of the time
- **NPV 88%**: If model says typical, it's correct 88% of the time

**Clinical Decision Rule**:
- If you want to **screen** (high sensitivity), accept lower specificity → catch all cases
- If you want **confirmation** (high specificity), accept lower sensitivity → minimize false alarms

---

### 3. **Feature Importance per Modality** (`modality_importance.png`)
**Purpose**: Shows which STRUCTURAL (sMRI) and FUNCTIONAL (fMRI) features are biomarkers

**What it shows**:
- **Left Panel**: Top sMRI features (structural brain anatomy biomarkers)
  - Desikan-Killiany cortical regions
  - ASEG subcortical structures
  - WMPARC white matter pathways
- **Right Panel**: Top fMRI features (functional connectivity biomarkers)
  - Network connectivity patterns

**Neuroscience Interpretation**:
- **sMRI biomarkers**: Structural differences in gray/white matter related to autism
- **fMRI biomarkers**: Functional connectivity abnormalities in neural networks
- **Combined power**: Model uses both anatomy AND function to classify

**Example insights**:
```
sMRI Top Features:
- Hippocampus volume → Memory/spatial processing
- Prefrontal cortex surface area → Executive function
- Corpus callosum connectivity → Interhemispheric communication

fMRI Top Features:
- Default Mode Network connectivity → Self-referential thinking
- Social Brain Network → Face/emotion processing
- Language network → Communication
```

---

### 4. **Brain Region Contribution** (`brain_regions.png`)
**Purpose**: Shows WHICH SPECIFIC BRAIN REGIONS are autism biomarkers

**What it shows**:
- Horizontal bar chart of top 20 sMRI brain regions
- **Color coding**:
  - 🔴 **Red** = Cortical regions (gray matter)
  - 🔵 **Blue** = Subcortical structures (deep brain)
  - 🟢 **Green** = White matter pathways

**Neuroscience Value**:
- **Dentate Gyrus** (red bar, high importance) → Hippocampus involvement in ASD
- **Amygdala** (blue bar) → Emotion/face processing abnormality
- **Corpus Callosum** (green bar) → Inter-brain hemisphere communication

**Examples of actual biomarkers**:
- **Superior Temporal Sulcus (STS)**: Biological motion perception
- **Fusiform Gyrus**: Face processing (often atypical in autism)
- **Amygdala**: Emotion recognition and social processing
- **Caudate Nucleus**: Reward/motivation system
- **Corpus Callosum**: Interhemispheric communication

**Interpretation**:
If your top regions align with known autism neurobiology (social brain, mirror neurons, etc.), the model has learned **valid neuroscience**, not just statistical patterns!

---

### 5. **Confusion Matrix** (Enhanced) (`confusion_matrix.png`)
**Purpose**: Clinical value—shows balance between missing cases and false alarms

**What it shows**:
```
              TD (Predicted)    ASD (Predicted)
TD (True)          TN ✓              FP ✗
ASD (True)         FN ✗              TP ✓
```

**Interpretation**:
- **TN (True Negative)**: Correctly identified typical development ✓
- **TP (True Positive)**: Correctly identified autism ✓
- **FP (False Positive)**: Wrongly said ASD (false alarm) ✗
- **FN (False Negative)**: Missed ASD case ✗

**Clinical trade-offs**:
- Many **FN** = **Missed diagnoses** (dangerous for clinical use)
- Many **FP** = **Unnecessary follow-up** (resource-intensive)

---

### 6. **ROC per Fold** (`roc_per_fold.png`)
**Purpose**: Shows consistency across cross-validation folds

**What it shows**:
- ROC curve for each of 5 folds
- Labels show AUC for each fold

**Interpretation**:
- All curves similar shape and position → Model is **stable** ✓
- Wide variation → Model **unreliable** (possibly overfitting) ✗

---

### 7. **Class Distribution per Fold** (`class_distribution.png`)
**Purpose**: Verifies proper stratification—no dataset bias

**What it shows**:
- Class counts (ASD vs TD) for train/val/test in each fold
- Stacked bar showing 50/50 split (or actual imbalance)

**Interpretation**:
- All splits have similar ratios → **stratified correctly** ✓
- One fold heavily imbalanced → **biased evaluation** ✗

---

### 8. **Training History** (`training_history.png`)
**Purpose**: Shows model stability and absence of overfitting

**What it shows**:
- Validation accuracy per fold (left)
- Test accuracy per fold (right)

**Interpretation**:
- Similar val/test accuracy → **good generalization** ✓
- Test << Val → **overfitting** ✗
- High variance across folds → **unstable model** ✗

---

## How to Interpret Results for a Neuroscience Publication

### Step 1: Check Generalization
- Open `site_performance.png`
- If all sites have accuracy within ±5%, you can confidently say: 
  > "Model generalizes across multiple imaging sites and scanner protocols, demonstrating applicability to real-world clinical settings."

### Step 2: Document Clinical Utility
- Open `clinical_metrics.png`
- Report sensitivity, specificity, PPV, NPV
- Discuss clinical implications:
  > "With sensitivity of 85%, the model misses 1 in 7 ASD cases. With specificity of 82%, it incorrectly flags 1 in 5 typical cases as ASD. This makes it suitable for screening (high sensitivity) but not diagnostic confirmation (moderate specificity)."

### Step 3: Identify Biomarkers
- Open `brain_regions.png` + `modality_importance.png`
- Map top regions to known autism neurobiology
- Write discussion section:
  > "The model identified structural abnormalities in the amygdala, hippocampus, and corpus callosum—regions consistently implicated in autism pathophysiology. Functional connectivity in the social brain network was also predictive, aligning with the social communication difficulties in ASD."

### Step 4: Support Claims
- Use `site_performance.png` → "Generalizes across sites"
- Use `brain_regions.png` → "Identifies known biomarkers"
- Use `clinical_metrics.png` → "Clinically balanced sensitivity/specificity"
- Use `roc_per_fold.png` → "Robust across cross-validation"

---

## Output File Locations

All figures are saved in: `/results/save_models/{ATLAS}_sMRI/with_ComBat/figures/`

```
figures/
├── training_history.png          ← Model stability
├── roc_curve.png                 ← Overall performance
├── roc_per_fold.png              ← Consistency check
├── confusion_matrix.png           ← Basic clinical metrics
├── clinical_metrics.png           ← Detailed clinical interpretation
├── feature_importance.png         ← Top 30 features overall
├── modality_importance.png        ← sMRI vs fMRI biomarkers
├── brain_regions.png              ← Specific brain regions
├── site_performance.png           ← No overfitting to scanners
├── class_distribution.png         ← Stratification verification
└── results.csv                    ← Numerical results
```

---

## Quick Publication Checklist

- [ ] All sites have similar accuracy? → Generalization proof ✓
- [ ] Sensitivity + Specificity both >80%? → Clinical utility ✓
- [ ] Top brain regions match known autism neurobiology? → Valid biomarkers ✓
- [ ] ROC curves consistent across folds? → Robust model ✓
- [ ] Class distribution balanced in splits? → Unbiased evaluation ✓
- [ ] Val/Test accuracy similar? → No overfitting ✓

---

## Example Publication-Ready Statement

> "We developed a multimodal neuroimaging classifier combining functional and structural MRI to identify autism spectrum disorder (ASD) with 82% accuracy (AUC=0.87). The model generalized across five independent imaging sites (site accuracy range: 79-85%), demonstrating applicability to diverse clinical settings. Biomarker analysis identified significant contributions from the social brain network (fMRI connectivity), amygdala volume, and corpus callosum integrity—regions consistently implicated in ASD neurobiology. With 85% sensitivity and 82% specificity, the model is suitable for screening but should be combined with clinical assessment for diagnostic confirmation."

---

**All figures designed to tell the story: "Our model is clinically useful, neurobiologically valid, and generalizable across sites."** ✓

