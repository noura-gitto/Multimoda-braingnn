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
