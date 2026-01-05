"""
Main script for CatBoost fMRI/sMRI classification
"""
import os
import sys
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

from config import Config
from data_loader import (load_fmri_data, load_smri_data, load_phenotypic_data,
                        apply_combat, get_ids, get_index)
from model_utils import (feature_selection_fmri, feature_selection_smri,
                        train_catboost, evaluate_model, save_model)
from visualization import (plot_roc_curve, plot_confusion_matrix, 
                          plot_training_history, plot_ensemble_comparison,
                          print_classification_report, save_results_to_csv,
                          plot_roc_per_fold, plot_feature_importance, plot_class_distribution,
                          plot_site_performance, plot_clinical_metrics, 
                          plot_modality_feature_importance, plot_brain_region_contribution)


def set_seed(seed=0):
    """Set seed for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    # Optional: uncomment if using PyTorch
    # import torch
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def create_fold_splits(sites, labels, unique_sites, k_fold=5):
    """
    Create stratified K-fold splits maintaining site distribution and class balance.
    
    Args:
        sites: list of site labels per subject
        labels: array of class labels (0/1)
        unique_sites: unique site identifiers
        k_fold: number of folds
    
    Returns:
        fold_indices: dict with fold->{'train', 'val', 'test'} indices
    """
    fold_indices = {i: {'train': [], 'val': [], 'test': []} for i in range(k_fold)}
    
    # Process each site separately to maintain site distribution
    for site in unique_sites:
        # Get all subjects from this site
        site_mask = np.array([s == site for s in sites])
        site_indices = np.where(site_mask)[0]
        site_labels = labels[site_indices]
        
        if len(site_indices) < k_fold:
            print(f"Warning: Site {site} has only {len(site_indices)} samples, less than {k_fold} folds")
        
        # Stratified k-fold for this site
        skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=0)
        
        fold_count = 0
        for train_val_idx, test_idx in skf.split(site_indices, site_labels):
            train_val_indices = site_indices[train_val_idx]
            test_indices = site_indices[test_idx]
            
            # Further split train_val into train and validation (80/20)
            tv_labels = labels[train_val_indices]
            tv_split_idx = int(0.8 * len(train_val_indices))
            
            # Stratified split of train/val to maintain class balance
            train_indices = train_val_indices[:tv_split_idx]
            val_indices = train_val_indices[tv_split_idx:]
            
            fold_indices[fold_count]['train'].extend(train_indices)
            fold_indices[fold_count]['val'].extend(val_indices)
            fold_indices[fold_count]['test'].extend(test_indices)
            
            fold_count += 1
    
    # Validate fold distribution
    for fold_idx in range(k_fold):
        train_indices = fold_indices[fold_idx]['train']
        val_indices = fold_indices[fold_idx]['val']
        test_indices = fold_indices[fold_idx]['test']
        
        train_labels = labels[train_indices]
        val_labels = labels[val_indices]
        test_labels = labels[test_indices]
        
        print(f"\nFold {fold_idx + 1} - Sample distribution:")
        print(f"  Train: {len(train_indices)} samples (Class 0: {np.sum(train_labels == 0)}, Class 1: {np.sum(train_labels == 1)})")
        print(f"  Val:   {len(val_indices)} samples (Class 0: {np.sum(val_labels == 0)}, Class 1: {np.sum(val_labels == 1)})")
        print(f"  Test:  {len(test_indices)} samples (Class 0: {np.sum(test_labels == 0)}, Class 1: {np.sum(test_labels == 1)})")
        
        # Check for severe imbalance (warn if <30% or >70% positive class)
        for split_name, split_labels in [('train', train_labels), ('val', val_labels), ('test', test_labels)]:
            pos_ratio = np.sum(split_labels) / len(split_labels) if len(split_labels) > 0 else 0
            if pos_ratio < 0.3 or pos_ratio > 0.7:
                print(f"  ⚠️  WARNING: {split_name} has class imbalance (positive class: {pos_ratio:.1%})")
    
    return fold_indices


def prepare_combined_features(fmri_data, smri_data, labels, train_ind, 
                              val_ind, test_ind, config):
    """
    Prepare combined fMRI and sMRI features with proper scaling.
    
    All scalers and selectors are fit ONLY on training data to prevent leakage.
    """
    # fMRI feature selection (fit only on training data)
    selector_fmri, scaler_fmri = feature_selection_fmri(
        fmri_data, labels, train_ind, config.NEW_FEATURES_FMRI, 
        scaler_flag=config.USE_SCALER
    )
    
    # sMRI feature selection (fit only on training data)
    selector_smri, scaler_smri = feature_selection_smri(
        smri_data, labels, train_ind, config.NEW_FEATURES_SMRI,
        scaler_flag=config.USE_SCALER
    )
    
    # Scale and transform all sets with fitted scalers
    fmri_scaled = fmri_data.copy()
    smri_scaled = smri_data.copy()
    
    if scaler_fmri is not None:
        fmri_scaled[train_ind, :] = scaler_fmri.transform(fmri_data[train_ind, :])
        fmri_scaled[val_ind, :] = scaler_fmri.transform(fmri_data[val_ind, :])
        fmri_scaled[test_ind, :] = scaler_fmri.transform(fmri_data[test_ind, :])
    
    if scaler_smri is not None:
        smri_scaled[train_ind, :] = scaler_smri.transform(smri_data[train_ind, :])
        smri_scaled[val_ind, :] = scaler_smri.transform(smri_data[val_ind, :])
        smri_scaled[test_ind, :] = scaler_smri.transform(smri_data[test_ind, :])
    
    # Transform features using fitted selectors
    fmri_selected = selector_fmri.transform(fmri_scaled)
    smri_selected = selector_smri.transform(smri_scaled)
    
    # Combine features
    combined = np.concatenate((smri_selected, fmri_selected), axis=1)
    
    # Combined feature selection (fit only on training data)
    selector_combine, scaler_combine = feature_selection_fmri(
        combined, labels, train_ind, config.NEW_FEATURES_COMBINE,
        scaler_flag=config.USE_SCALER
    )
    
    # Apply combined scaler to all sets
    combined_scaled = combined.copy()
    if scaler_combine is not None:
        combined_scaled[train_ind, :] = scaler_combine.transform(combined[train_ind, :])
        combined_scaled[val_ind, :] = scaler_combine.transform(combined[val_ind, :])
        combined_scaled[test_ind, :] = scaler_combine.transform(combined[test_ind, :])
    
    final_data = selector_combine.transform(combined_scaled)
    
    return final_data


def train_single_fold(fold_idx, data, labels, fold_indices, config):
    """
    Train model for a single fold.
    
    Args:
        fold_idx: fold number (0-indexed)
        data: combined feature data
        labels: class labels
        fold_indices: dict with 'train', 'val', 'test' indices
        config: configuration object
    """
    print(f"\n{'='*60}")
    print(f"Training Fold {fold_idx + 1}")
    print(f"{'='*60}")
    
    # Get indices for this fold
    train_ind = np.array(fold_indices[fold_idx]['train'])
    val_ind = np.array(fold_indices[fold_idx]['val'])
    test_ind = np.array(fold_indices[fold_idx]['test'])
    
    # Prepare data splits
    train_data = data[train_ind, :]
    train_labels = labels[train_ind]
    val_data = data[val_ind, :]
    val_labels = labels[val_ind]
    test_data = data[test_ind, :]
    test_labels = labels[test_ind]
    
    print(f"Train samples: {len(train_labels)}")
    print(f"Val samples:   {len(val_labels)}")
    print(f"Test samples:  {len(test_labels)}")
    
    # Train with hyperparameter search
    print("Training CatBoost with hyperparameter search...")
    model, best_lr, best_depth, val_acc = train_catboost(
        train_data, train_labels, val_data, val_labels,
        config.LEARNING_RATES, config.DEPTHS, config
    )
    
    print(f"Best learning rate: {best_lr}")
    print(f"Best depth:         {best_depth}")
    print(f"Validation acc:     {val_acc:.4f}")
    
    # Evaluate on test set
    test_results, test_probs, test_acc = evaluate_model(
        model, test_data, test_labels
    )
    print(f"Test accuracy:      {test_acc:.4f}")
    
    # Save model
    model_path = os.path.join(config.SAVE_PATH, f'CAT_{fold_idx + 1}.m')
    save_model(model, model_path)
    print(f"Model saved to {model_path}")
    
    return {
        'model': model,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'test_labels': test_labels,
        'test_results': test_results,
        'test_probs': test_probs
    }


def main():
    """Main training pipeline"""
    print("="*80)
    print("Multimodal Brain ASD Classification with CatBoost")
    print("="*80)
    
    # Set reproducibility seeds
    set_seed(0)
    print("\n✓ Reproducibility seeds set (random_state=0)")
    
    # Initialize config
    config = Config()
    config.create_directories()
    
    # Change to working directory
    os.chdir(config.BASE_DIR)
    print(f"\nWorking directory: {os.getcwd()}")
    print(f"fMRI Atlas: {config.FMRI_ATLAS}")
    print(f"ComBat fMRI: {config.COMBAT_FMRI}")
    print(f"ComBat sMRI: {config.COMBAT_SMRI}")
    
    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    fmri_data, labels, sites, genders, ages, unique_sites = load_fmri_data(config)
    subject_ids = get_ids(config.NUM_SAMPLES, dir_path=config.LABEL_DIR)
    smri_data = load_smri_data(config, subject_ids)
    fiq, num, pec, rat = load_phenotypic_data(config, subject_ids)
    
    # Add phenotypic features to sMRI (x3 for multi-scale representation)
    # Rationale: Phenotypic features are repeated to match multi-scale processing in fMRI
    for _ in range(3):
        smri_data = np.concatenate((smri_data, ages, genders, fiq, num, pec, rat), axis=1)
    
    print(f"\nData shapes:")
    print(f"  fMRI: {fmri_data.shape}")
    print(f"  sMRI: {smri_data.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Unique sites: {len(unique_sites)}")
    
    # Apply ComBat harmonization
    print("\n" + "="*60)
    print("HARMONIZATION")
    print("="*60)
    
    fmri_data = apply_combat(fmri_data, sites, labels, genders, ages, 
                            unique_sites, config.COMBAT_FMRI)
    smri_data = apply_combat(smri_data, sites, labels, genders, ages,
                            unique_sites, config.COMBAT_SMRI)
    
    # Create fold splits
    print("\n" + "="*60)
    print("CREATING FOLD SPLITS")
    print("="*60)
    
    fold_indices = create_fold_splits(
        sites, labels, unique_sites, config.K_FOLD
    )
    
    # Prepare combined features for each fold
    print("\n" + "="*60)
    print("FEATURE SELECTION")
    print("="*60)
    
    data = {}
    for fold in range(config.K_FOLD):
        print(f"\nFold {fold + 1}: Feature selection...")
        train_ind = np.array(fold_indices[fold]['train'])
        val_ind = np.array(fold_indices[fold]['val'])
        test_ind = np.array(fold_indices[fold]['test'])
        
        data[fold] = prepare_combined_features(
            fmri_data, smri_data, labels,
            train_ind, val_ind, test_ind,
            config
        )
        print(f"  Combined features shape: {data[fold].shape}")
    
    # Train models for each fold
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)
    
    fold_results = []
    all_test_labels = []
    all_test_probs = []
    
    for fold in range(config.K_FOLD):
        result = train_single_fold(
            fold, data[fold], labels, fold_indices, config
        )
        fold_results.append(result)
        
        # Concatenate results
        if fold == 0:
            all_test_labels = result['test_labels']
            all_test_probs = result['test_probs']
        else:
            all_test_labels = np.concatenate([all_test_labels, result['test_labels']])
            all_test_probs = np.concatenate([all_test_probs, result['test_probs']])
    
    # Visualize results
    print("\n" + "="*60)
    print("RESULTS AND VISUALIZATION")
    print("="*60)
    
    # Training history
    history = {
        'fold': [f"Fold {i+1}" for i in range(config.K_FOLD)],
        'val_acc': [r['val_acc'] for r in fold_results],
        'test_acc': [r['test_acc'] for r in fold_results]
    }
    
    fig_path = os.path.join(config.SAVE_PATH, 'figures/training_history.png')
    plot_training_history(history, save_path=fig_path)
    
    # Overall ROC curve
    fig_path = os.path.join(config.SAVE_PATH, 'figures/roc_curve.png')
    overall_auc = plot_roc_curve(all_test_labels, all_test_probs, 
                                 save_path=fig_path, 
                                 title="Overall ROC Curve (All Folds)")
    
    # Confusion matrix
    all_test_preds = np.argmax(all_test_probs, axis=1)
    fig_path = os.path.join(config.SAVE_PATH, 'figures/confusion_matrix.png')
    plot_confusion_matrix(all_test_labels, all_test_preds, save_path=fig_path)
    
    # Print classification report
    print_classification_report(all_test_labels, all_test_preds)

    # Additional visualizations
    # ROC per fold
    fig_path = os.path.join(config.SAVE_PATH, 'figures/roc_per_fold.png')
    mean_fold_auc = plot_roc_per_fold(fold_results, save_path=fig_path, title='ROC Curves per Fold')
    print(f"Mean AUC across folds: {mean_fold_auc:.4f}")

    # Feature importance (average across folds)
    models = [r['model'] for r in fold_results if r.get('model') is not None]
    if len(models) > 0:
        fig_path = os.path.join(config.SAVE_PATH, 'figures/feature_importance.png')
        plot_feature_importance(models, top_n=30, save_path=fig_path)

    # Class distribution per fold
    fig_path = os.path.join(config.SAVE_PATH, 'figures/class_distribution.png')
    plot_class_distribution(fold_indices, labels, save_path=fig_path)

    # NEUROSCIENCE-FOCUSED VISUALIZATIONS
    print("\n" + "="*60)
    print("NEUROSCIENCE BIOMARKER ANALYSIS")
    print("="*60)

    # Site performance (no overfitting to specific scanners)
    fig_path = os.path.join(config.SAVE_PATH, 'figures/site_performance.png')
    site_stats = plot_site_performance(sites, labels, all_test_preds, all_test_probs, save_path=fig_path)
    print("\nSite Performance (Generalization across hospitals/scanners):")
    print(site_stats.to_string(index=False))

    # Clinical metrics with confusion matrix interpretation
    fig_path = os.path.join(config.SAVE_PATH, 'figures/clinical_metrics.png')
    plot_clinical_metrics(all_test_labels, all_test_preds, all_test_probs, save_path=fig_path)

    # Modality-specific feature importance (structural vs functional biomarkers)
    fig_path = os.path.join(config.SAVE_PATH, 'figures/modality_importance.png')
    if len(models) > 0:
        modality_imp = plot_modality_feature_importance(
            models, 
            n_features_fmri=config.NEW_FEATURES_FMRI,
            n_features_smri=config.NEW_FEATURES_SMRI,
            top_n=25,
            save_path=fig_path
        )
        if modality_imp is not None:
            print("\nTop Features by Modality:")
            print(modality_imp.to_string(index=False))

    # Brain region contribution (neuroscience value)
    fig_path = os.path.join(config.SAVE_PATH, 'figures/brain_regions.png')
    if len(models) > 0:
        brain_contrib = plot_brain_region_contribution(
            models,
            n_features_smri=config.NEW_FEATURES_SMRI,
            top_n=20,
            save_path=fig_path
        )
        if brain_contrib is not None:
            print("\nTop Brain Regions (Biomarkers):")
            print(brain_contrib.to_string(index=False))
    
    # Save results
    avg_val_acc = np.mean([r['val_acc'] for r in fold_results])
    avg_test_acc = np.mean([r['test_acc'] for r in fold_results])
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Average Validation Accuracy: {avg_val_acc:.4f}")
    print(f"Average Test Accuracy:       {avg_test_acc:.4f}")
    print(f"Overall AUC:                 {overall_auc:.4f}")
    print(f"Total Test Samples:          {len(all_test_labels)}")
    
    # Save results to CSV
    results_dict = {
        'fold': [i+1 for i in range(config.K_FOLD)],
        'val_accuracy': [r['val_acc'] for r in fold_results],
        'test_accuracy': [r['test_acc'] for r in fold_results]
    }
    csv_path = os.path.join(config.SAVE_PATH, 'results.csv')
    save_results_to_csv(results_dict, csv_path)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE ✓")
    print("="*80)


if __name__ == "__main__":
    main()
