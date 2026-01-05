# 📊 Data Leakage Analysis - Complete Package

**Status:** ✅ COMPLETE (6 files, ready to use)  
**Created:** January 5, 2026  
**Severity:** 🔴 CRITICAL - Affects publication credibility

---

## 📑 Documentation Files (Read These First)

### 1. **REVIEW_FINDINGS.md** ⭐ START HERE
- Visual summary with diagrams
- Problem diagnosis in plain language  
- Expected accuracy changes
- 15-minute read
- **Best for:** Quick understanding of the issue

### 2. **QUICK_FIX_GUIDE.md** ⚡ ACTION STEPS
- What to do right now
- Running instructions
- Expected results interpretation
- Per-site performance analysis
- 10-minute read  
- **Best for:** Getting started immediately

### 3. **DATA_LEAKAGE_FIX.md** 🔬 TECHNICAL ANALYSIS
- Detailed issue breakdown
- Root cause explanation
- ComBat theory and practice
- Site overfitting evidence
- 20-minute read
- **Best for:** Understanding the science

### 4. **DATA_LEAKAGE_COMPLETE_ANALYSIS.md** 📋 FULL REFERENCE
- Comprehensive overview
- All key points in one place
- Troubleshooting guide
- Publication impact analysis
- **Best for:** Reference during implementation

---

## 💻 Executable Scripts (Ready to Run)

### 5. **main_fixed.py** 🔧 RECOMMENDED
```bash
python main_fixed.py
```

**What it does:**
- Fixed 5-fold cross-validation with proper isolation
- Applies ComBat only to training data of each fold
- Includes balanced accuracy and macro F1 metrics
- Shows per-site performance table
- Honest, reproducible results

**Expected output:**
- Accuracy likely 65-85% (down from 94%)
- More uniform per-site performance
- Fair metrics for imbalanced classes
- CSV file with detailed results

**Duration:** 20-30 minutes

---

### 6. **main_loso.py** 🏆 GOLD STANDARD
```bash
python main_loso.py
```

**What it does:**
- Leave-One-Site-Out cross-validation
- Trains on 19 sites, tests on 1
- Repeats for all 20 sites
- TRUE generalization test across hospitals

**Expected output:**
- Per-site accuracy table
- Mean accuracy ± std across sites
- Interpretation guidance
- CSV file with site-wise results

**Duration:** 45 minutes - 2 hours (depends on compute)

**Critical interpretation:**
- LOSO > 70% → Real biomarkers ✓
- LOSO 60-70% → Moderate overfitting ⚠️
- LOSO < 60% → Heavy overfitting ✗

---

### 7. **main_unimodal.py** 🔍 DEBUG TOOL
```bash
python main_unimodal.py
```

**What it does:**
- Trains fMRI-only model
- Trains sMRI-only model
- Compares which modality has scanner bias

**Expected output:**
- Per-modality accuracy breakdown
- Feature importance by modality
- Comparison table
- Diagnostic summary

**Duration:** 30-45 minutes

**Critical interpretation:**
- sMRI >> fMRI → sMRI has scanner bias
- sMRI ≈ fMRI → Both modalities OK
- fMRI >> sMRI → fMRI more robust

---

## 🎯 Quick Decision Tree

```
START HERE
    ↓
Read REVIEW_FINDINGS.md (15 min)
    ↓
Question: How much time do you have?
    ├─ < 1 hour?
    │  ├─ Read QUICK_FIX_GUIDE.md
    │  └─ Run main_fixed.py only
    │
    ├─ 1-2 hours?
    │  ├─ Read DATA_LEAKAGE_FIX.md
    │  ├─ Run main_fixed.py
    │  └─ Run main_loso.py
    │
    └─ > 2 hours?
       ├─ Read all documentation
       ├─ Run main_fixed.py
       ├─ Run main_loso.py
       └─ Run main_unimodal.py

After running scripts:
    ↓
Check results in generated CSV files
    ↓
Compare to original (CatBoost.log)
    ↓
Decide on publication strategy
```

---

## 📊 Which Script for Which Question?

| Question | Script | Read |
|---|---|---|
| Is my pipeline correct? | main_fixed.py | QUICK_FIX_GUIDE.md |
| Does my model generalize? | main_loso.py | DATA_LEAKAGE_FIX.md |
| Which modality is problematic? | main_unimodal.py | QUICK_FIX_GUIDE.md |
| How bad is the leakage? | main_fixed.py | REVIEW_FINDINGS.md |
| What do I change in my paper? | All three | DATA_LEAKAGE_COMPLETE_ANALYSIS.md |

---

## ✅ Validation Checklist

Before running, verify:

- [ ] Python environment matches original `main.py`
- [ ] All data files present in `data/` folder
- [ ] Config.py hasn't changed
- [ ] At least 4GB free disk space for results
- [ ] 2+ hours of compute time available (especially LOSO)

---

## 📈 Expected Results

### Accuracy Changes
```
Original pipeline:     94% ← INFLATED (leakage)
main_fixed.py:        65-85% ← HONEST (no leakage)
main_loso.py:         50-80% ← TRUE (cross-site)
main_unimodal.py:     Varies by modality
```

### Per-Site Performance
```
Original:
  NYU: 52%
  STANFORD: 24% ← Varies wildly
  CMU: 45%
  ...
  
Fixed:
  NYU: ~65% ← More uniform
  STANFORD: ~68%
  CMU: ~62%
```

---

## 🚨 If Results Show Heavy Overfitting

If `main_loso.py` shows accuracy < 60%, consider:

1. **Stronger harmonization**
   - Try ComBat-GAM (nonlinear)
   - Or FreeSurfer's built-in harmonization

2. **Feature selection improvements**
   - Use site-stratified RFE
   - Remove features correlated with site

3. **Model complexity reduction**
   - Simpler CatBoost parameters
   - Fewer features overall

4. **fMRI-only results**
   - If sMRI dominates, use fMRI-only model
   - sMRI likely contains pure scanner bias

---

## 📝 Documentation Overview

| File | Purpose | Read Time | Best For |
|---|---|---|---|
| REVIEW_FINDINGS.md | Visual summary | 15 min | Quick overview |
| QUICK_FIX_GUIDE.md | Action steps | 10 min | Getting started |
| DATA_LEAKAGE_FIX.md | Technical details | 20 min | Understanding science |
| DATA_LEAKAGE_COMPLETE_ANALYSIS.md | Full reference | 30 min | Implementation |
| main_fixed.py | Fixed pipeline | N/A | Run directly |
| main_loso.py | LOSO evaluation | N/A | Run directly |
| main_unimodal.py | Modality comparison | N/A | Run directly |

---

## 🔑 Key Findings Summary

```
PROBLEM: Model achieved 94% accuracy globally but only
         24-52% per-site, indicating site overfitting

CAUSE:   ComBat harmonization applied before fold splits,
         leaking test set information into training

IMPACT:  - Inflated accuracy metrics
         - False biomarker claims
         - Poor generalization across hospitals
         - Manuscript would fail peer review

SOLUTION: Three new scripts implement proper methodology:
         1. main_fixed.py   → Honest 5-fold CV
         2. main_loso.py    → Gold standard LOSO CV
         3. main_unimodal.py → Debug modality contributions

NEXT:    Read documentation, run scripts, compare results,
         update manuscript with honest metrics
```

---

## 🎓 What You'll Learn

By working through this analysis, you'll understand:

1. **Data leakage mechanisms** - How information flows between splits
2. **Cross-validation pitfalls** - Why site-based data needs special handling
3. **Multisite brain imaging** - Gold standards (LOSO) for generalization
4. **Honest reporting** - Fair metrics for imbalanced data
5. **Overfitting detection** - How to spot when models memorize noise

---

## ⏱️ Time Estimate

| Activity | Time |
|---|---|
| Read REVIEW_FINDINGS.md | 15 min |
| Read QUICK_FIX_GUIDE.md | 10 min |
| Run main_fixed.py | 20-30 min |
| Run main_loso.py | 45 min - 2 hrs |
| Run main_unimodal.py | 30-45 min |
| Analyze & compare results | 30 min |
| Update manuscript | 2-4 hrs |
| **TOTAL** | **4-8 hours** |

---

## 🎯 Success Criteria

After completing this analysis, you should be able to:

- [ ] Explain why original results were inflated
- [ ] Interpret LOSO accuracy correctly
- [ ] Understand per-site performance variation
- [ ] Identify which modality contains scanner bias
- [ ] Report honest, reproducible results
- [ ] Defend methodology to peer reviewers

---

## 📧 Questions?

All documentation is self-contained. If specific errors occur:

1. Check Python syntax: `python -m py_compile script.py`
2. Review error message against script comments
3. Verify data paths in config.py
4. Check available disk space and compute resources

---

## 🏁 Next Steps (Right Now)

1. ✅ You are here (reading this file)
2. → Open **REVIEW_FINDINGS.md**
3. → Open **QUICK_FIX_GUIDE.md**  
4. → Run `python main_fixed.py`
5. → Run `python main_loso.py` (important!)
6. → Compare results
7. → Update your manuscript

---

**STATUS:** ✅ Ready to execute  
**COMPLEXITY:** Medium (straightforward fixes)  
**TIME TO RESOLUTION:** 4-8 hours total  
**IMPACT:** Critical (protects manuscript credibility)

Start with REVIEW_FINDINGS.md →
