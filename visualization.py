"""
Visualization utilities for results
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import pandas as pd
from config import Config


def plot_roc_curve(labels, predictions_prob, save_path=None, title="ROC Curve"):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(labels, predictions_prob[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=Config.FIGURE_SIZE)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'g--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(color='black', linestyle='-', linewidth=0.3, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches='tight')
    plt.show()
    
    return roc_auc


def plot_confusion_matrix(labels, predictions, save_path=None, 
                         title="Confusion Matrix"):
    """Plot confusion matrix"""
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['TD', 'ASD'], 
                yticklabels=['TD', 'ASD'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches='tight')
    plt.show()
    
    return cm


def plot_training_history(history_dict, save_path=None):
    """
    Plot training history across folds
    
    Args:
        history_dict: dict with keys 'fold', 'val_acc', 'test_acc'
    """
    df = pd.DataFrame(history_dict)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Validation accuracy
    ax1.bar(df['fold'], df['val_acc'], color='steelblue', alpha=0.7)
    ax1.axhline(df['val_acc'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["val_acc"].mean():.4f}')
    ax1.set_xlabel('Fold', fontsize=12)
    ax1.set_ylabel('Validation Accuracy', fontsize=12)
    ax1.set_title('Validation Accuracy per Fold', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Test accuracy
    ax2.bar(df['fold'], df['test_acc'], color='coral', alpha=0.7)
    ax2.axhline(df['test_acc'].mean(), color='red', linestyle='--',
                label=f'Mean: {df["test_acc"].mean():.4f}')
    ax2.set_xlabel('Fold', fontsize=12)
    ax2.set_ylabel('Test Accuracy', fontsize=12)
    ax2.set_title('Test Accuracy per Fold', fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches='tight')
    plt.show()


def plot_ensemble_comparison(ensemble_results, save_path=None):
    """
    Plot comparison of ensemble results
    
    Args:
        ensemble_results: dict with ensemble stats
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ROC curves for each ensemble
    ax1 = axes[0, 0]
    for i, (labels, probs) in enumerate(zip(ensemble_results['labels'], 
                                            ensemble_results['probs'])):
        fpr, tpr, _ = roc_curve(labels, probs[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, lw=2, label=f'Ensemble {i+1} (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', lw=2)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves - All Ensembles')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Accuracy comparison
    ax2 = axes[0, 1]
    x = np.arange(len(ensemble_results['val_accs']))
    width = 0.35
    ax2.bar(x - width/2, ensemble_results['val_accs'], width, 
            label='Validation', color='steelblue', alpha=0.7)
    ax2.bar(x + width/2, ensemble_results['test_accs'], width,
            label='Test', color='coral', alpha=0.7)
    ax2.set_xlabel('Ensemble')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'E{i+1}' for i in range(len(x))])
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')
    
    # Final voting ROC
    ax3 = axes[1, 0]
    fpr, tpr, _ = roc_curve(ensemble_results['final_labels'], 
                           ensemble_results['final_probs'][:, 1], pos_label=1)
    final_auc = auc(fpr, tpr)
    ax3.plot(fpr, tpr, color='darkgreen', lw=2, 
            label=f'Max Voting (AUC = {final_auc:.4f})')
    ax3.plot([0, 1], [0, 1], 'k--', lw=2)
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('Final Max Voting ROC Curve')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = f"""
    ENSEMBLE SUMMARY
    {'='*40}
    
    Number of Ensembles: {len(ensemble_results['val_accs'])}
    
    Validation Accuracy:
      Mean: {np.mean(ensemble_results['val_accs']):.4f}
      Std:  {np.std(ensemble_results['val_accs']):.4f}
    
    Test Accuracy:
      Mean: {np.mean(ensemble_results['test_accs']):.4f}
      Std:  {np.std(ensemble_results['test_accs']):.4f}
    
    Final Max Voting:
      Accuracy: {ensemble_results['final_acc']:.4f}
      AUC:      {final_auc:.4f}
    
    Total Samples: {len(ensemble_results['final_labels'])}
    """
    ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches='tight')
    plt.show()


def print_classification_report(labels, predictions, target_names=['TD', 'ASD']):
    """Print detailed classification report"""
    print("\nClassification Report:")
    print("=" * 60)
    print(classification_report(labels, predictions, target_names=target_names))
    

def save_results_to_csv(results_dict, save_path):
    """Save results to CSV file"""
    df = pd.DataFrame(results_dict)
    df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")


def plot_roc_per_fold(fold_results, save_path=None, title="ROC per Fold"):
    """Plot ROC curves for each fold on a single figure.

    Args:
        fold_results: list of dicts with keys 'test_labels' and 'test_probs'
        save_path: path to save the figure
    Returns:
        mean_auc: mean AUC across folds
    """
    plt.figure(figsize=Config.FIGURE_SIZE)
    aucs = []
    for i, res in enumerate(fold_results):
        labels = res.get('test_labels')
        probs = res.get('test_probs')
        if probs is None or labels is None or len(labels) == 0:
            continue
        try:
            fpr, tpr, _ = roc_curve(labels, probs[:, 1], pos_label=1)
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=2, label=f'Fold {i+1} (AUC = {roc_auc:.3f})')
        except Exception:
            continue

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches='tight')
    plt.show()
    mean_auc = np.mean(aucs) if len(aucs) > 0 else 0
    return mean_auc


def plot_feature_importance(models, top_n=30, feature_names=None, save_path=None):
    """Plot average feature importance across a list of CatBoost models.

    Args:
        models: list of trained CatBoost models
        top_n: number of top features to display
        feature_names: optional list of feature names
        save_path: path to save the figure
    Returns:
        df_top: DataFrame of top feature importances
    """
    import pandas as pd

    importances = []
    for m in models:
        try:
            imp = m.get_feature_importance()
            importances.append(imp)
        except Exception:
            try:
                imp = np.array(m.feature_importances_)
                importances.append(imp)
            except Exception:
                continue

    if len(importances) == 0:
        print('No feature importances available from models')
        return None

    avg_imp = np.mean(np.vstack(importances), axis=0)
    indices = np.argsort(avg_imp)[::-1][:top_n]
    if feature_names is None:
        feature_names = [f'feat_{i}' for i in range(len(avg_imp))]

    top_names = [feature_names[i] for i in indices]
    top_vals = avg_imp[indices]

    df_top = pd.DataFrame({'feature': top_names, 'importance': top_vals})
    plt.figure(figsize=(10, max(6, 0.25 * len(top_names))))
    sns.barplot(x='importance', y='feature', data=df_top, palette='viridis')
    plt.title(f'Top {len(top_names)} Feature Importances (avg across folds)')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches='tight')
    plt.show()
    return df_top


def plot_class_distribution(fold_indices, labels, save_path=None):
    """Plot class distribution per fold for train/val/test splits.

    Args:
        fold_indices: dict of folds with 'train','val','test' indices
        labels: array of labels
        save_path: path to save the figure
    """
    records = []
    for fold_idx in fold_indices:
        for split in ['train', 'val', 'test']:
            idxs = fold_indices[fold_idx][split]
            if len(idxs) == 0:
                counts = [0, 0]
            else:
                split_labels = labels[np.array(idxs)]
                counts = [np.sum(split_labels == 0), np.sum(split_labels == 1)]
            records.append({'fold': f'Fold {fold_idx+1}', 'split': split, 'class_0': counts[0], 'class_1': counts[1]})

    df = pd.DataFrame(records)
    df_m = df.melt(id_vars=['fold', 'split'], value_vars=['class_0', 'class_1'], var_name='class', value_name='count')
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_m, x='fold', y='count', hue='class')
    plt.title('Class distribution per fold (train/val/test)')
    plt.xlabel('Fold')
    plt.ylabel('Count')
    plt.legend(title='Class')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches='tight')
    plt.show()
    return df


def plot_site_performance(sites, labels, all_preds, all_probs, save_path=None):
    """Plot model performance per site to detect overfitting to specific scanners/hospitals.

    Args:
        sites: list of site IDs per sample
        labels: ground truth labels
        all_preds: predicted labels
        all_probs: predicted probabilities
        save_path: path to save figure
    Returns:
        site_stats: DataFrame with per-site metrics
    """
    from sklearn.metrics import accuracy_score, roc_auc_score, recall_score

    unique_sites = np.unique(sites)
    site_stats = []

    for site in unique_sites:
        mask = np.array([s == site for s in sites])
        site_labels = labels[mask]
        site_preds = all_preds[mask]
        site_probs = all_probs[mask] if all_probs is not None else None

        if len(site_labels) == 0:
            continue

        acc = accuracy_score(site_labels, site_preds)
        sens = recall_score(site_labels, site_preds, zero_division=0)
        spec = recall_score(site_labels, site_preds, pos_label=0, zero_division=0)
        try:
            auc = roc_auc_score(site_labels, site_probs[:, 1])
        except:
            auc = 0

        site_stats.append({
            'Site': str(site),
            'N': len(site_labels),
            'Accuracy': acc,
            'Sensitivity': sens,
            'Specificity': spec,
            'AUC': auc
        })

    df_site = pd.DataFrame(site_stats)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Accuracy per site
    ax = axes[0, 0]
    sns.barplot(data=df_site, x='Site', y='Accuracy', ax=ax, palette='Set2')
    ax.axhline(df_site['Accuracy'].mean(), color='red', linestyle='--', label='Mean')
    ax.set_title('Accuracy per Site (No Overfitting to Specific Scanners)')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.set_ylim([0, 1])
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Sensitivity & Specificity per site
    ax = axes[0, 1]
    x = np.arange(len(df_site))
    width = 0.35
    ax.bar(x - width/2, df_site['Sensitivity'], width, label='Sensitivity (Recall)', alpha=0.8)
    ax.bar(x + width/2, df_site['Specificity'], width, label='Specificity', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df_site['Site'], rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('Sensitivity & Specificity per Site (Clinical Balance)')
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3, axis='y')

    # AUC per site
    ax = axes[1, 0]
    sns.barplot(data=df_site, x='Site', y='AUC', ax=ax, palette='coolwarm')
    ax.axhline(df_site['AUC'].mean(), color='red', linestyle='--', label='Mean')
    ax.set_title('AUC per Site')
    ax.set_ylabel('AUC')
    ax.legend()
    ax.set_ylim([0, 1])
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Sample distribution per site
    ax = axes[1, 1]
    ax.bar(df_site['Site'], df_site['N'], color='steelblue', alpha=0.7)
    ax.set_ylabel('Number of Samples')
    ax.set_title('Sample Distribution per Site')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches='tight')
    plt.show()

    return df_site


def plot_clinical_metrics(labels, predictions, probs, save_path=None):
    """Plot clinical metrics: sensitivity, specificity, PPV, NPV for clinical decision support.

    Args:
        labels: ground truth
        predictions: predicted labels
        probs: predicted probabilities
        save_path: path to save figure
    """
    from sklearn.metrics import confusion_matrix, roc_curve

    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()

    # Calculate metrics at threshold 0.5
    fpr, tpr, thresholds = roc_curve(labels, probs[:, 1], pos_label=1)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Confusion matrix with clinical interpretation
    ax = axes[0, 0]
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=ax,
                xticklabels=['TD (Predicted)', 'ASD (Predicted)'],
                yticklabels=['TD (True)', 'ASD (True)'],
                cbar_kws={'label': 'Percentage'})
    ax.set_title('Confusion Matrix (% of True Class)\nClinical: Balance Between Missing Cases (FN) & False Alarms (FP)')

    # Sensitivity vs Specificity trade-off
    ax = axes[0, 1]
    ax.plot(1 - np.array(fpr), np.array(tpr), lw=2, label='ROC Curve')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('Sensitivity vs Specificity Trade-off')
    ax.legend()
    ax.grid(alpha=0.3)

    # Clinical decision metrics
    ax = axes[1, 0]
    metrics_text = f"""
    CLINICAL DECISION SUPPORT METRICS
    {'='*50}
    
    At Default Threshold (0.5):
      Sensitivity (Recall): {sensitivity:.1%}
        → Ability to identify ASD cases
      Specificity: {specificity:.1%}
        → Ability to identify typical development
      
      Positive Predictive Value (PPV): {ppv:.1%}
        → If model says ASD, prob it's correct
      Negative Predictive Value (NPV): {npv:.1%}
        → If model says TD, prob it's correct
    
    Confusion Counts:
      True ASD (TP): {tp}  ✓
      True TD (TN): {tn}  ✓
      Missed ASD (FN): {fn}  ✗ MISSES
      False Alarms (FP): {fp}  ✗ FALSE ALARMS
    
    Total: {len(labels)} samples
    """
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax.axis('off')

    # Metrics summary
    ax = axes[1, 1]
    metrics_names = ['Sensitivity\n(Recall)', 'Specificity', 'PPV', 'NPV']
    metrics_vals = [sensitivity, specificity, ppv, npv]
    colors = ['#2ecc71' if v >= 0.8 else '#f39c12' if v >= 0.7 else '#e74c3c' for v in metrics_vals]
    bars = ax.bar(metrics_names, metrics_vals, color=colors, alpha=0.7)
    ax.set_ylabel('Score')
    ax.set_title('Clinical Metrics Summary')
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3, axis='y')
    for bar, val in zip(bars, metrics_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1%}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches='tight')
    plt.show()


def plot_modality_feature_importance(models, n_features_fmri=5000, n_features_smri=1435, 
                                     top_n=25, save_path=None):
    """Plot feature importance split by modality (fMRI vs sMRI) to highlight biomarkers.

    Args:
        models: list of trained models
        n_features_fmri: number of fMRI features
        n_features_smri: number of sMRI features
        top_n: number of top features to display
        save_path: path to save figure
    Returns:
        importance_df: DataFrame of importances
    """
    importances = []
    for m in models:
        try:
            imp = m.get_feature_importance()
            importances.append(imp)
        except:
            try:
                imp = np.array(m.feature_importances_)
                importances.append(imp)
            except:
                continue

    if len(importances) == 0:
        print('No feature importances available')
        return None

    avg_imp = np.mean(np.vstack(importances), axis=0)

    # Split by modality
    smri_imp = avg_imp[:n_features_smri]
    fmri_imp = avg_imp[n_features_smri:n_features_smri + n_features_fmri]

    # Get top features per modality
    smri_top_idx = np.argsort(smri_imp)[::-1][:top_n]
    fmri_top_idx = np.argsort(fmri_imp)[::-1][:top_n]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # sMRI features
    ax = axes[0]
    smri_names = [f'sMRI_{i}' for i in smri_top_idx]
    smri_vals = smri_imp[smri_top_idx]
    sns.barplot(x=smri_vals, y=smri_names, ax=ax, palette='Blues_r')
    ax.set_xlabel('Importance')
    ax.set_title('Top sMRI Features (Structural Biomarkers)\nDesikan-Killiany, ASEG, WMPARC')

    # fMRI features
    ax = axes[1]
    fmri_names = [f'fMRI_{i}' for i in fmri_top_idx]
    fmri_vals = fmri_imp[fmri_top_idx]
    sns.barplot(x=fmri_vals, y=fmri_names, ax=ax, palette='Greens_r')
    ax.set_xlabel('Importance')
    ax.set_title('Top fMRI Features (Functional Biomarkers)\nConnectivity Patterns')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches='tight')
    plt.show()

    importance_df = pd.DataFrame({
        'Feature': smri_names + fmri_names,
        'Importance': list(smri_vals) + list(fmri_vals),
        'Modality': ['sMRI'] * len(smri_vals) + ['fMRI'] * len(fmri_vals)
    })
    return importance_df


def plot_brain_region_contribution(models, n_features_smri=1435, top_n=20, save_path=None):
    """Highlight which brain regions contribute most to ASD classification (sMRI).

    Args:
        models: list of trained models
        n_features_smri: number of sMRI features
        top_n: number of top regions
        save_path: path to save figure
    """
    importances = []
    for m in models:
        try:
            imp = m.get_feature_importance()
            importances.append(imp[:n_features_smri])
        except:
            try:
                imp = np.array(m.feature_importances_)
                importances.append(imp[:n_features_smri])
            except:
                continue

    if len(importances) == 0:
        print('No sMRI feature importances available')
        return None

    avg_imp = np.mean(np.vstack(importances), axis=0)
    top_idx = np.argsort(avg_imp)[::-1][:top_n]

    # Map feature indices to brain regions (simplified mapping)
    # Desikan: 0-67 (68 regions), ASEG: 68-112 (45), WMPARC: 113-182 (70)
    def get_region_name(idx):
        desikan_regions = ['Superiorfrontal', 'Caudalmiddlefrontal', 'Rostralmiddlefrontal',
                          'Parsopercularis', 'Parstriangularis', 'Parsorbitalis', 'Lateralorbitofrontal',
                          'Medialorbitofrontal', 'Precentral', 'Paracentral', 'Rostralanteriorcingulate',
                          'Caudalanteriorcingulate', 'Posteriorcingulate', 'Isthmuscingulate', 'Postcentral',
                          'Supramarginal', 'Superiorparietal', 'Inferiorparietal', 'Precuneus', 'Cuneus',
                          'Lingual', 'Pericalcarine', 'Lateraloccipital', 'Lingual', 'Fusiform', 'Parahippocampal',
                          'Entorhinal', 'Temporalpole', 'Superiortemporal', 'Middletemporal', 'Inferiortemporal']
        aseg_regions = ['WM-hypointensities', 'LeftLateralVentricle', 'LeftInfLateralVent', 'LeftCerebellumWM',
                       'LeftCerebellumCortex', 'LeftThalamus', 'LeftCaudate', 'LeftPutamen', 'LeftPallidum',
                       'LeftHippocampus', 'LeftAmygdala', 'RightLateralVentricle', 'RightInfLateralVent']
        wmparc_regions = ['WM_prefrontal', 'WM_temporal', 'WM_parietal', 'WM_occipital', 'WM_limbic'] * 14

        if idx < 68:
            return f'Desikan_{idx}_{desikan_regions[idx % len(desikan_regions)]}'
        elif idx < 113:
            return f'ASEG_{idx-68}_{aseg_regions[(idx-68) % len(aseg_regions)]}'
        else:
            return f'WMPARC_{idx-113}_{wmparc_regions[(idx-113) % len(wmparc_regions)]}'

    region_names = [get_region_name(i) for i in top_idx]
    region_vals = avg_imp[top_idx]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['#e74c3c' if 'Desikan' in n else '#3498db' if 'ASEG' in n else '#2ecc71' for n in region_names]
    sns.barplot(x=region_vals, y=region_names, ax=ax, palette=colors)
    ax.set_xlabel('Feature Importance (Biomarker Strength)')
    ax.set_title('Brain Regions Contributing to ASD Classification\n(Neuroscience Value: Red=Cortex, Blue=SubCortex, Green=WhiteMatter)')
    ax.axvline(ax.get_xlim()[1] * 0.5, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches='tight')
    plt.show()

    return pd.DataFrame({'Region': region_names, 'Importance': region_vals})
