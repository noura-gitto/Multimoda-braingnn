"""
Unimodal Experiments: Test sMRI-only vs fMRI-only models

If sMRI achieves much higher accuracy than fMRI, the sMRI features likely 
contain site-specific scanner signatures rather than ASD biomarkers.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from config import Config
from data_loader import (load_fmri_data, load_smri_data, load_phenotypic_data,
                        apply_combat, get_ids)
from model_utils import (feature_selection_fmri, feature_selection_smri,
                        train_catboost, evaluate_model)


def set_seed(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)


def train_unimodal_model(data_type, data, labels, sites, genders, ages, 
                         unique_sites, config):
    """
    Train model using only one modality (fMRI or sMRI).
    
    Args:
        data_type: 'fMRI' or 'sMRI'
        data: feature matrix for this modality
        labels, sites, genders, ages: metadata
        unique_sites: list of unique sites
        config: Config object
    
    Returns:
        results_df: DataFrame with fold-wise metrics
    """
    print(f"\n{'='*70}")
    print(f"Training {data_type}-Only Model")
    print(f"{'='*70}")
    
    # Create fold splits
    fold_indices = {i: {'train': [], 'val': [], 'test': []} for i in range(config.K_FOLD)}
    
    for site in unique_sites:
        site_mask = np.array([s == site for s in sites])
        site_indices = np.where(site_mask)[0]
        site_labels = labels[site_indices]
        
        skf = StratifiedKFold(n_splits=config.K_FOLD, shuffle=True, random_state=0)
        fold_count = 0
        
        for train_val_idx, test_idx in skf.split(site_indices, site_labels):
            train_val_indices = site_indices[train_val_idx]
            test_indices = site_indices[test_idx]
            
            tv_labels = labels[train_val_indices]
            tv_split_idx = int(0.8 * len(train_val_indices))
            
            train_indices = train_val_indices[:tv_split_idx]
            val_indices = train_val_indices[tv_split_idx:]
            
            fold_indices[fold_count]['train'].extend(train_indices)
            fold_indices[fold_count]['val'].extend(val_indices)
            fold_indices[fold_count]['test'].extend(test_indices)
            
            fold_count += 1
    
    # Train on each fold
    fold_results = []
    
    for fold in range(config.K_FOLD):
        print(f"\n  Fold {fold + 1}...")
        
        train_ind = np.array(fold_indices[fold]['train'])
        val_ind = np.array(fold_indices[fold]['val'])
        test_ind = np.array(fold_indices[fold]['test'])
        
        # Feature selection fit only on training data
        if data_type == 'fMRI':
            n_features = config.NEW_FEATURES_FMRI
            selector, scaler = feature_selection_fmri(
                data, labels, train_ind, n_features, scaler_flag=config.USE_SCALER
            )
        else:  # sMRI
            n_features = config.NEW_FEATURES_SMRI
            selector, scaler = feature_selection_smri(
                data, labels, train_ind, n_features, scaler_flag=config.USE_SCALER
            )
        
        # Scale data using fitted scaler
        data_scaled = data.copy()
        if scaler is not None:
            data_scaled[train_ind, :] = scaler.transform(data[train_ind, :])
            data_scaled[val_ind, :] = scaler.transform(data[val_ind, :])
            data_scaled[test_ind, :] = scaler.transform(data[test_ind, :])
        
        # Select features
        data_selected = selector.transform(data_scaled)
        
        # Train model
        train_data = data_selected[train_ind, :]
        train_labels = labels[train_ind]
        val_data = data_selected[val_ind, :]
        val_labels = labels[val_ind]
        test_data = data_selected[test_ind, :]
        test_labels = labels[test_ind]
        
        model, best_lr, best_depth, val_acc = train_catboost(
            train_data, train_labels, val_data, val_labels,
            config.LEARNING_RATES, config.DEPTHS, config
        )
        
        # Evaluate
        test_preds = model.predict(test_data)
        test_probs = model.predict_proba(test_data)
        test_acc = accuracy_score(test_labels, test_preds)
        balanced_acc = balanced_accuracy_score(test_labels, test_preds)
        macro_f1 = f1_score(test_labels, test_preds, average='macro')
        
        try:
            auc = roc_auc_score(test_labels, test_probs[:, 1])
        except:
            auc = np.nan
        
        print(f"    Acc: {test_acc:.4f}, Bal.Acc: {balanced_acc:.4f}, F1: {macro_f1:.4f}, AUC: {auc:.4f}")
        
        fold_results.append({
            'fold': fold + 1,
            'accuracy': test_acc,
            'balanced_accuracy': balanced_acc,
            'macro_f1': macro_f1,
            'auc': auc
        })
    
    return pd.DataFrame(fold_results)


def main():
    """Run unimodal comparison"""
    print("="*80)
    print("Unimodal Modality Comparison: fMRI vs sMRI")
    print("="*80)
    
    set_seed(0)
    
    # Load config and data
    config = Config()
    config.create_directories()
    os.chdir(config.BASE_DIR)
    
    print(f"\nWorking directory: {os.getcwd()}")
    
    print("\nLoading data...")
    fmri_data, labels, sites, genders, ages, unique_sites = load_fmri_data(config)
    subject_ids = get_ids(config.NUM_SAMPLES, dir_path=config.LABEL_DIR)
    smri_data = load_smri_data(config, subject_ids)
    fiq, num, pec, rat = load_phenotypic_data(config, subject_ids)
    
    # Add phenotypic features to sMRI
    for _ in range(3):
        smri_data = np.concatenate((smri_data, ages, genders, fiq, num, pec, rat), axis=1)
    
    print(f"  fMRI: {fmri_data.shape}")
    print(f"  sMRI: {smri_data.shape}")
    
    # Apply ComBat harmonization
    print("\nApplying ComBat harmonization...")
    fmri_data = apply_combat(fmri_data, sites, labels, genders, ages,
                             unique_sites, config.COMBAT_FMRI)
    smri_data = apply_combat(smri_data, sites, labels, genders, ages,
                             unique_sites, config.COMBAT_SMRI)
    
    # Train unimodal models
    fmri_results = train_unimodal_model(
        'fMRI', fmri_data, labels, sites, genders, ages, unique_sites, config
    )
    
    smri_results = train_unimodal_model(
        'sMRI', smri_data, labels, sites, genders, ages, unique_sites, config
    )
    
    # Summary
    print(f"\n{'='*80}")
    print("UNIMODAL COMPARISON RESULTS")
    print(f"{'='*80}")
    
    print("\nfMRI-Only Model:")
    print(fmri_results.to_string(index=False))
    print(f"\n  Mean Accuracy:       {fmri_results['accuracy'].mean():.4f}")
    print(f"  Mean Balanced Acc:   {fmri_results['balanced_accuracy'].mean():.4f}")
    print(f"  Mean Macro F1:       {fmri_results['macro_f1'].mean():.4f}")
    print(f"  Mean AUC:            {fmri_results['auc'].mean():.4f}")
    
    print("\n\nsMRI-Only Model:")
    print(smri_results.to_string(index=False))
    print(f"\n  Mean Accuracy:       {smri_results['accuracy'].mean():.4f}")
    print(f"  Mean Balanced Acc:   {smri_results['balanced_accuracy'].mean():.4f}")
    print(f"  Mean Macro F1:       {smri_results['macro_f1'].mean():.4f}")
    print(f"  Mean AUC:            {smri_results['auc'].mean():.4f}")
    
    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON:")
    print(f"{'='*80}")
    fmri_acc = fmri_results['accuracy'].mean()
    smri_acc = smri_results['accuracy'].mean()
    diff = abs(fmri_acc - smri_acc)
    
    print(f"\nAccuracy Difference: {diff:.4f}")
    if diff > 0.15:  # >15% difference
        better = "sMRI" if smri_acc > fmri_acc else "fMRI"
        print(f"\n⚠️  WARNING: Large modality performance gap!")
        print(f"   {better} significantly outperforms the other modality.")
        print(f"   This suggests one modality contains site-specific scanner signature")
        print(f"   rather than genuine ASD biomarkers.")
    else:
        print(f"\n✓ Both modalities perform similarly")
        print(f"  This is a good sign for genuine biomarker discovery")
    
    # Save results
    comparison_df = pd.DataFrame({
        'Modality': ['fMRI', 'sMRI'],
        'Mean Accuracy': [fmri_acc, smri_acc],
        'Mean Balanced Acc': [
            fmri_results['balanced_accuracy'].mean(),
            smri_results['balanced_accuracy'].mean()
        ],
        'Mean Macro F1': [
            fmri_results['macro_f1'].mean(),
            smri_results['macro_f1'].mean()
        ],
        'Mean AUC': [
            fmri_results['auc'].mean(),
            smri_results['auc'].mean()
        ]
    })
    
    csv_path = os.path.join(config.SAVE_PATH, 'unimodal_comparison.csv')
    comparison_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
