# Quick Reference: Running the Fixed Pipeline

## The Problem in One Sentence
**Your model learned to identify the scanner/site, not ASD biomarkers** - evidenced by 94% global accuracy but only 24-52% per-site accuracy.

---

## The Root Cause
ComBat harmonization was applied **before** fold splitting, leaking test set information into training.

---

## Your Options

### Option 1: Quick Sanity Check (5 minutes)
```bash
# Run the fixed 5-fold CV to see if accuracy drops
python main_fixed.py > fixed_results.log

# Compare results:
tail -20 CatBoost.log      # Original (94% accuracy)
tail -20 fixed_results.log # Fixed version (accuracy will likely drop)
```

If accuracy drops significantly → **Leakage was happening**.

---

### Option 2: Full Diagnostic Suite (30 minutes)

```bash
# 1. Fixed standard CV
python main_fixed.py

# 2. Leave-One-Site-Out (most important!)
python main_loso.py

# 3. Unimodal analysis
python main_unimodal.py
```

Then analyze:
1. **Did accuracy drop in main_fixed.py?** → Confirms leakage
2. **What's the LOSO accuracy?** → True generalization
3. **Does sMRI >> fMRI in main_unimodal.py?** → sMRI contains site bias

---

### Option 3: Just Use the Fixed Main (Recommended)

Replace your `main.py` with `main_fixed.py`:

```bash
# Back up original
cp main.py main.py.backup

# Use fixed version
cp main_fixed.py main.py

# Run as usual
python main.py
```

The fixed version:
- ✓ Applies ComBat per-fold (no leakage)
- ✓ Uses balanced accuracy & macro F1 metrics
- ✓ Shows per-site performance
- ✓ Same interface as original

---

## What to Expect

### Original Pipeline (`main.py`)
```
Global Accuracy: 94%
Per-site range: 24%-93%
Imbalance: No correction
```

### Fixed Pipeline (`main_fixed.py`)
```
Global Accuracy: (likely 65-80%)
Per-site range: (more uniform)
Imbalance: Corrected with balanced metrics
```

**Your accuracy will probably drop. This is GOOD - it means you've removed artificial inflation.**

---

## Interpreting LOSO Results

Leave-One-Site-Out tests if your model generalizes across hospitals.

```python
# If LOSO results show:

LOSO Accuracy ≈ 70-85%  → ✓ Model learned genuine biomarkers
LOSO Accuracy ≈ 50-65%  → ⚠️ Site effects significant
LOSO Accuracy < 50%     → ✗ Model only learns scanner patterns
```

If LOSO is much worse than 5-fold CV, your model is **site-overfitted**.

---

## Interpreting Unimodal Results

fMRI-only vs sMRI-only models reveal which modality contains site bias.

```
If sMRI >> fMRI accuracy:
  → White matter volumes contain scanner-specific bias
  → ComBat didn't fully remove site effects
  → Consider downweighting sMRI features

If fMRI ≈ sMRI accuracy:
  → Both modalities contribute genuine signal
  → Model is more trustworthy

If fMRI >> sMRI accuracy:
  → fMRI is more robust (optimal result)
```

---

## Code Changes Made

### main_fixed.py
- Moved ComBat application inside the fold loop
- Added balanced accuracy and macro F1 metrics
- Added per-site performance table
- Otherwise identical to main.py

### main_loso.py
- Implements Leave-One-Site-Out cross-validation
- Trains on 19 sites, tests on 1 (repeats for all 20)
- Shows true cross-site generalization

### main_unimodal.py
- Tests fMRI-only and sMRI-only models
- Compares performance to identify problematic modality
- Helps debug which features contain site bias

---

## Next Steps Based on Results

### If LOSO accuracy is good (>70%):
- Your model likely learned real biomarkers ✓
- Use LOSO results in your papers (more credible)
- Publish with confidence in generalization

### If LOSO accuracy is poor (<60%):
- Heavy site overfitting detected
- Action items:
  1. Use LOSO as primary metric (don't report inflated 5-fold)
  2. Try stronger harmonization (ComBat-GAM, Freesurfer harmonization)
  3. Focus on fMRI features (likely more site-robust)
  4. Consider reducing model complexity (prevent overfitting to site patterns)

### If sMRI >> fMRI in unimodal tests:
- White matter features likely contain scanner bias
- Recommend:
  1. Use fMRI-only for main results
  2. Treat sMRI as supporting evidence only
  3. Investigate which sMRI features drive site differences

---

## Files Created

```
main_fixed.py              ← Fixed 5-fold CV (drop-in replacement)
main_loso.py              ← Leave-One-Site-Out CV
main_unimodal.py          ← fMRI vs sMRI comparison
DATA_LEAKAGE_FIX.md       ← Detailed technical analysis
QUICK_FIX_GUIDE.md        ← This file
```

---

## One-Liner to Test Everything

```bash
python main_fixed.py && python main_loso.py && python main_unimodal.py
```

Then check the logs to compare.

---

## Key Metrics to Compare

### Original vs Fixed (5-fold CV)

| Metric | Original | Fixed | Interpretation |
|---|---|---|---|
| Global Accuracy | 94% | ??? | Leakage removed |
| Balanced Accuracy | Not reported | ??? | True fair metric |
| Per-site range | 24-93% | ??? | Generalization |

### LOSO Results

| Site | Accuracy | Notes |
|---|---|---|
| NYU | ?? | (largest site - most data) |
| STANFORD | ?? | (was 24% before) |
| ... | ... | Check all 20 sites |
| Mean | ??? | Compare to 5-fold mean |

### Unimodal Comparison

| Modality | Accuracy | Notes |
|---|---|---|
| fMRI-only | ??? | Baseline |
| sMRI-only | ??? | If higher → suspect |
| Combined | ??? | Your original result |

---

## Contact/Questions

Key questions to ask yourself:

1. **Is my per-site performance balanced?** 
   - If some sites have 80%+ and others <40%, you're overfitting sites

2. **Does LOSO accuracy justify my findings?**
   - LOSO is the "true" accuracy for multisite data

3. **Which modality drives the result?**
   - If one modality dominates with site-specific features, investigate

4. **Are my reported metrics honest?**
   - Use balanced accuracy/F1 instead of accuracy for imbalanced data

---

## Bottom Line

Your original pipeline had **data leakage** and **site overfitting**. The three new scripts will:

1. ✓ Remove the leakage (`main_fixed.py`)
2. ✓ Reveal true generalization (`main_loso.py`)
3. ✓ Debug which modality is problematic (`main_unimodal.py`)

**Run all three, compare results, and let the data tell you the truth.**
