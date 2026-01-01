"""
Main script for CatBoost fMRI/sMRI classification
"""
import os
import sys
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
                          print_classification_report, save_results_to_csv)


def create_fold_splits(sites, labels, unique_sites, k_fold=5):
    """Create stratified K-fold splits maintaining site distribution"""
    dist_train = {str(i + 1): [] for i in range(k_fold)}
    dist_validation = {str(i + 1): [] for i in range(k_fold)}
    dist_test = {str(i + 1): [] for i in range(k_fold)}
    
    for each_site in unique_sites:
        index_site = get_index(sites, each_site)
        label = np.array([int(labels[int(idx)]) for idx in index_site])
        
        # Stratified K-Fold
        sfolder = StratifiedKFold(n_splits=k_fold, random_state=0, shuffle=True)
        group = 0
        
        for train, validation in sfolder.split(index_site, label):
            for i in train:
                dist_train[str(group + 1)].append(index_site[i])
            for j in validation:
                dist_validation[str(group + 1)].append(index_site[j])
            group += 1
        
        # Create test sets (rotating validation sets)
        group = 0
        for train, validation in sfolder.split(index_site, label):
            if group == 0:
                for j in validation:
                    dist_test[str(group + k_fold)].append(index_site[j])
                    dist_train[str(group + k_fold)].remove(index_site[j])
            else:
                for j in validation:
                    dist_test[str(group)].append(index_site[j])
                    dist_train[str(group)].remove(index_site[j])
            group += 1
    
    return dist_train, dist_validation, dist_test


def prepare_combined_features(fmri_data, smri_data, labels, train_ind, 
                              val_ind, test_ind, config):
    """Prepare combined fMRI and sMRI features with proper scaling"""
    # fMRI feature selection
    selector_fmri, scaler_fmri, fmri_scaled = feature_selection_fmri(
        fmri_data, labels, train_ind, config.NEW_FEATURES_FMRI, 
        scaler_flag=config.USE_SCALER
    )
    
    # sMRI feature selection
    selector_smri, scaler_smri, smri_scaled = feature_selection_smri(
        smri_data, labels, train_ind, config.NEW_FEATURES_SMRI,
        scaler_flag=config.USE_SCALER
    )
    
    # Apply scalers to validation and test
    if scaler_fmri is not None:
        fmri_scaled[val_ind, :] = scaler_fmri.transform(fmri_data[val_ind, :])
        fmri_scaled[test_ind, :] = scaler_fmri.transform(fmri_data[test_ind, :])
    
    if scaler_smri is not None:
        smri_scaled[val_ind, :] = scaler_smri.transform(smri_data[val_ind, :])
        smri_scaled[test_ind, :] = scaler_smri.transform(smri_data[test_ind, :])
    
    # Transform features
    fmri_selected = selector_fmri.transform(fmri_scaled)
    smri_selected = selector_smri.transform(smri_scaled)
    
    # Combine features
    combined = np.concatenate((smri_selected, fmri_selected), axis=1)
    
    # Combined feature selection
    selector_combine, scaler_combine, combined_scaled = feature_selection_fmri(
        combined, labels, train_ind, config.NEW_FEATURES_COMBINE,
        scaler_flag=config.USE_SCALER
    )
    
    # Apply combined scaler
    if scaler_combine is not None:
        combined_scaled[val_ind, :] = scaler_combine.transform(combined[val_ind, :])
        combined_scaled[test_ind, :] = scaler_combine.transform(combined[test_ind, :])
    
    final_data = selector_combine.transform(combined_scaled)
    
    return final_data


def train_single_fold(fold_idx, data, labels, dist_train, dist_val, dist_test, config):
    """Train model for a single fold"""
    print(f"\n{'='*60}")
    print(f"Training Fold {fold_idx + 1}")
    print(f"{'='*60}")
    
    # Prepare data splits
    train_data = data[dist_train[str(fold_idx + 1)], :]
    train_labels = labels[dist_train[str(fold_idx + 1)]]
    val_data = data[dist_val[str(fold_idx + 1)], :]
    val_labels = labels[dist_val[str(fold_idx + 1)]]
    test_data = data[dist_test[str(fold_idx + 1)], :]
    test_labels = labels[dist_test[str(fold_idx + 1)]]
    
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
    print("CatBoost fMRI/sMRI Classification")
    print("="*80)
    
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
    
    # Add phenotypic features to sMRI (repeat 3 times as in original)
    for _ in range(3):
        smri_data = np.concatenate((smri_data, ages, genders, fiq, num, pec, rat), axis=1)
    
    print(f"\nData shapes:")
    print(f"  fMRI: {fmri_data.shape}")
    print(f"  sMRI: {smri_data.shape}")
    print(f"  Labels: {labels.shape}")
    
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
    
    dist_train, dist_val, dist_test = create_fold_splits(
        sites, labels, unique_sites, config.K_FOLD
    )
    
    # Prepare combined features for each fold
    print("\n" + "="*60)
    print("FEATURE SELECTION")
    print("="*60)
    
    data = {}
    for fold in range(config.K_FOLD):
        print(f"\nFold {fold + 1}: Feature selection...")
        data[str(fold + 1)] = prepare_combined_features(
            fmri_data, smri_data, labels,
            dist_train[str(fold + 1)],
            dist_val[str(fold + 1)],
            dist_test[str(fold + 1)],
            config
        )
        print(f"  Combined features shape: {data[str(fold + 1)].shape}")
    
    # Train models for each fold
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)
    
    fold_results = []
    all_test_labels = []
    all_test_probs = []
    
    for fold in range(config.K_FOLD):
        result = train_single_fold(
            fold, data[str(fold + 1)], labels,
            dist_train, dist_val, dist_test, config
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
    print("TRAINING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
