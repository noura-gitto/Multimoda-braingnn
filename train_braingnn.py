"""
Training Pipeline for BrainGNN-Multimodal
Includes data loading, training, validation, and evaluation
"""

import os
import sys
import json
import numpy as np
import scipy.io as scio
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import random
import math

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Scikit-learn imports
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, 
    confusion_matrix, classification_report, f1_score
)
from sklearn.preprocessing import StandardScaler

# Import our model
from braingnn_multimodal import BrainGNNMultimodal, create_model

# Import visualization utilities
from visualization import (
    plot_roc_curve, plot_confusion_matrix, plot_training_history,
    plot_roc_per_fold, plot_site_performance, plot_clinical_metrics,
    print_classification_report, save_results_to_csv
)


# ============================================================================
# Setup Logging
# ============================================================================

def setup_logging(save_dir: str) -> logging.Logger:
    """Setup logging configuration"""
    log_file = os.path.join(save_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ============================================================================
# Random Seed Setting
# ============================================================================

def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Dataset Class
# ============================================================================

class ABIDEDataset(Dataset):
    """
    PyTorch Dataset for ABIDE data
    """
    def __init__(self, 
                 fmri_data: np.ndarray,
                 smri_data: np.ndarray,
                 labels: np.ndarray,
                 sites: np.ndarray,
                 ages: np.ndarray,
                 genders: np.ndarray,
                 fiqs: np.ndarray,
                 indices: List[int],
                 site_to_idx: Dict[str, int],
                 augment: bool = False):
        """
        Args:
            fmri_data: (num_samples, num_nodes, num_nodes)
            smri_data: (num_samples, smri_dim)
            labels: (num_samples,)
            sites: (num_samples,) - site names
            ages: (num_samples,)
            genders: (num_samples,)
            fiqs: (num_samples,)
            indices: List of indices to include in this dataset
            site_to_idx: Mapping from site names to indices
            augment: Whether to apply data augmentation
        """
        self.fmri_data = fmri_data[indices]
        self.smri_data = smri_data[indices]
        self.labels = labels[indices]
        self.sites = np.array([site_to_idx[sites[i]] for i in indices])
        self.ages = ages[indices]
        self.genders = genders[indices]
        self.fiqs = fiqs[indices]
        self.augment = augment
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        fmri = self.fmri_data[idx].copy()
        smri = self.smri_data[idx].copy()
        
        # Data augmentation
        if self.augment:
            # fMRI augmentation
            if random.random() < 0.5:
                # Add Gaussian noise
                fmri += np.random.normal(0, 0.01, fmri.shape)
            
            if random.random() < 0.3:
                # Random edge dropout
                mask = np.random.binomial(1, 0.9, fmri.shape)
                fmri = fmri * mask
            
            # sMRI augmentation
            if random.random() < 0.5:
                # Add Gaussian noise
                smri += np.random.normal(0, 0.05, smri.shape)
        
        return {
            'fmri': torch.FloatTensor(fmri),
            'smri': torch.FloatTensor(smri),
            'label': torch.LongTensor([self.labels[idx]])[0],
            'site': torch.LongTensor([self.sites[idx]])[0],
            'age': torch.FloatTensor([self.ages[idx]]),
            'gender': torch.LongTensor([self.genders[idx]])[0],
            'fiq': torch.FloatTensor([self.fiqs[idx]])
        }


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_fmri_data(root_path: str, subject_IDs: List[str], 
                   num_nodes: int, logger: logging.Logger) -> np.ndarray:
    """
    Load fMRI connectivity matrices
    
    Returns:
        fmri_data: (num_subjects, num_nodes, num_nodes)
    """
    logger.info(f"Loading fMRI data from {root_path}")
    
    num_subjects = len(subject_IDs)
    fmri_data = np.zeros((num_subjects, num_nodes, num_nodes))
    
    for i, subject_id in enumerate(tqdm(subject_IDs, desc="Loading fMRI")):
        try:
            mat_file = os.path.join(root_path, f"{subject_id}.mat")
            data = scio.loadmat(mat_file)
            connectivity = data['connectivity']
            fmri_data[i] = connectivity
        except Exception as e:
            logger.warning(f"Error loading {subject_id}: {e}")
            # Use zero matrix if loading fails
            fmri_data[i] = np.zeros((num_nodes, num_nodes))
    
    logger.info(f"Loaded fMRI data shape: {fmri_data.shape}")
    return fmri_data


def load_smri_data(freesurfer_path: str, subject_IDs: List[str], 
                   logger: logging.Logger) -> np.ndarray:
    """
    Load sMRI features from FreeSurfer stats files
    
    Returns:
        smri_data: (num_subjects, total_features)
    """
    logger.info(f"Loading sMRI data from {freesurfer_path}")
    
    # Feature configurations
    parcellations = {
        'desikan_killiany': {
            'files': ['lh.aparc.stats', 'rh.aparc.stats'],
            'features': ['NumVert', 'SurfArea', 'GrayVol', 'ThickAvg', 'ThickStd', 
                        'MeanCurv', 'GausCurv', 'FoldInd', 'CurvInd'],
            'skiprows': list(range(61))
        },
        'aseg': {
            'files': ['aseg.stats'],
            'features': ['Number of Voxels', 'Volume', 'Intensity normMean', 
                        'Itensity normStdDev', 'Intensity normMin', 
                        'Intensity normMax', 'Intensity normRange'],
            'skiprows': list(range(79))
        }
    }
    
    all_features = []
    
    for i, subject_id in enumerate(tqdm(subject_IDs, desc="Loading sMRI")):
        subject_features = []
        
        try:
            subject_dir = os.path.join(freesurfer_path, subject_id)
            
            # Load Desikan-Killiany features
            for hemi_file in parcellations['desikan_killiany']['files']:
                file_path = os.path.join(subject_dir, hemi_file)
                if os.path.exists(file_path):
                    df = pd.read_table(file_path, sep='\s+', 
                                     skiprows=parcellations['desikan_killiany']['skiprows'])
                    for feat in parcellations['desikan_killiany']['features']:
                        if feat in df.columns:
                            subject_features.extend(df[feat].values)
            
            # Load aseg features
            aseg_file = os.path.join(subject_dir, 'aseg.stats')
            if os.path.exists(aseg_file):
                df = pd.read_table(aseg_file, sep='\s+', 
                                 skiprows=parcellations['aseg']['skiprows'])
                for feat in parcellations['aseg']['features']:
                    if feat in df.columns:
                        subject_features.extend(df[feat].values)
            
        except Exception as e:
            logger.warning(f"Error loading sMRI for {subject_id}: {e}")
        
        all_features.append(subject_features)
    
    # Convert to numpy array and handle variable lengths
    max_len = max(len(f) for f in all_features)
    smri_data = np.zeros((len(subject_IDs), max_len))
    
    for i, features in enumerate(all_features):
        smri_data[i, :len(features)] = features
    
    logger.info(f"Loaded sMRI data shape: {smri_data.shape}")
    return smri_data


def load_phenotypic_data(pheno_dir: str, num_subjects: int, 
                        logger: logging.Logger) -> Dict[str, np.ndarray]:
    """
    Load phenotypic data (labels, age, gender, FIQ, sites)
    
    Returns:
        Dictionary with phenotypic data
    """
    logger.info(f"Loading phenotypic data from {pheno_dir}")
    
    # Load labels
    labels = scio.loadmat(os.path.join(pheno_dir, 'ABIDE_label_871.mat'))['label'][0]
    
    # Load ages
    ages = scio.loadmat(os.path.join(pheno_dir, 'ages.mat'))['ages'].flatten()
    ages = np.array([float(str(a).replace(' ', '')) for a in ages])
    
    # Load genders
    genders = scio.loadmat(os.path.join(pheno_dir, 'genders.mat'))['genders'].flatten()
    genders = np.array([int(g) for g in genders])
    
    # Load FIQ
    try:
        fiqs = scio.loadmat(os.path.join(pheno_dir, 'FIQS.mat'))['FIQS'].flatten()
        fiqs = np.array([float(str(f).replace(' ', '')) if str(f).strip() else 100.0 for f in fiqs])
    except:
        logger.warning("FIQ data not found, using default values")
        fiqs = np.ones(num_subjects) * 100.0
    
    # Load sites
    sites = scio.loadmat(os.path.join(pheno_dir, 'sites.mat'))['sites']
    sites = np.array([str(s).replace(' ', '') for s in sites])
    
    # Load subject IDs
    subject_IDs = np.genfromtxt(os.path.join(pheno_dir, 'subject_IDs.txt'), dtype=str)
    
    logger.info(f"Loaded phenotypic data for {num_subjects} subjects")
    logger.info(f"Labels distribution: ASD={np.sum(labels==1)}, TD={np.sum(labels==0)}")
    logger.info(f"Number of unique sites: {len(np.unique(sites))}")
    
    return {
        'labels': labels,
        'ages': ages,
        'genders': genders,
        'fiqs': fiqs,
        'sites': sites,
        'subject_IDs': subject_IDs.tolist()
    }


# ============================================================================
# Cross-Validation Split Function
# ============================================================================

def create_site_aware_splits(sites: np.ndarray, labels: np.ndarray, 
                            k_fold: int = 5, random_state: int = 42,
                            logger: Optional[logging.Logger] = None) -> Dict:
    """
    Create site-aware stratified k-fold splits
    
    Returns:
        Dictionary with train, val, test indices for each fold
    """
    if logger:
        logger.info(f"Creating {k_fold}-fold site-aware splits")
    
    unique_sites = np.unique(sites)
    num_samples = len(labels)
    
    fold_splits = {}
    
    for fold in range(k_fold):
        train_indices = []
        val_indices = []
        test_indices = []
        
        for site in unique_sites:
            site_mask = sites == site
            site_indices = np.where(site_mask)[0]
            site_labels = labels[site_indices]
            
            # Skip if too few samples
            if len(site_indices) < k_fold:
                # Add all to train
                train_indices.extend(site_indices.tolist())
                continue
            
            # Stratified k-fold for this site
            skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=random_state)
            
            for fold_idx, (train_val, test) in enumerate(skf.split(site_indices, site_labels)):
                if fold_idx == fold:
                    # Further split train_val into train and val
                    train_val_indices = site_indices[train_val]
                    train_val_labels = site_labels[train_val]
                    
                    if len(train_val_indices) >= 2:
                        val_size = max(1, len(train_val_indices) // 4)
                        skf_inner = StratifiedKFold(n_splits=min(4, len(train_val_indices)), 
                                                   shuffle=True, random_state=random_state)
                        for inner_idx, (train, val) in enumerate(skf_inner.split(train_val_indices, train_val_labels)):
                            if inner_idx == 0:
                                train_indices.extend(train_val_indices[train].tolist())
                                val_indices.extend(train_val_indices[val].tolist())
                                break
                    else:
                        train_indices.extend(train_val_indices.tolist())
                    
                    test_indices.extend(site_indices[test].tolist())
                    break
        
        fold_splits[fold] = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
        
        if logger:
            logger.info(f"Fold {fold+1}: Train={len(train_indices)}, "
                       f"Val={len(val_indices)}, Test={len(test_indices)}")
    
    return fold_splits


# ============================================================================
# Training Functions
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss

class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with weighted components
    """
    def __init__(self, lambda_cls: float = 5.0, lambda_site: float = 0.1, 
                 lambda_age: float = 0.05, lambda_reg: float = 0.001):
        super(MultiTaskLoss, self).__init__()
        self.lambda_cls = lambda_cls
        self.lambda_site = lambda_site
        self.lambda_age = lambda_age
        self.lambda_reg = lambda_reg
        
        self.cls_criterion = FocalLoss(label_smoothing=0.1)
        self.site_criterion = nn.CrossEntropyLoss()
        self.age_criterion = nn.MSELoss()
    
    def forward(self, class_logits, site_logits, age_pred, 
                labels, sites, ages, model):
        # Classification loss
        cls_loss = self.cls_criterion(class_logits, labels)
        
        # Site prediction loss (for domain adaptation)
        site_loss = self.site_criterion(site_logits, sites)
        
        # Age regression loss (for deconfounding)
        age_loss = self.age_criterion(age_pred.squeeze(), ages.squeeze())
        
        # L2 regularization
        l2_reg = torch.tensor(0., device=class_logits.device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        
        # Total loss
        total_loss = (self.lambda_cls * cls_loss + 
                     self.lambda_site * site_loss + 
                     self.lambda_age * age_loss + 
                     self.lambda_reg * l2_reg)
        
        return total_loss, {
            'cls_loss': cls_loss.item(),
            'site_loss': site_loss.item(),
            'age_loss': age_loss.item(),
            'l2_reg': l2_reg.item()
        }


def train_epoch(model: nn.Module, dataloader: DataLoader, 
                criterion: MultiTaskLoss, optimizer: optim.Optimizer,
                device: torch.device, epoch: int) -> Dict:
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    loss_components = {'cls_loss': 0, 'site_loss': 0, 'age_loss': 0, 'l2_reg': 0}
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch in pbar:
        # Move data to device
        fmri = batch['fmri'].to(device)
        smri = batch['smri'].to(device)
        labels = batch['label'].to(device)
        sites = batch['site'].to(device)
        ages = batch['age'].to(device)
        genders = batch['gender'].to(device)
        fiqs = batch['fiq'].to(device)
        
        # Forward pass
        class_logits, site_logits, age_pred, _ = model(
            fmri, smri, sites, ages, genders, fiqs
        )
        
        # Compute loss
        loss, loss_dict = criterion(
            class_logits, site_logits, age_pred, 
            labels, sites, ages, model
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += loss_dict[key]
        
        preds = torch.argmax(class_logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    metrics = {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        **{k: v / len(dataloader) for k, v in loss_components.items()}
    }
    
    return metrics


def evaluate(model: nn.Module, dataloader: DataLoader, 
            criterion: MultiTaskLoss, device: torch.device,
            phase: str = 'Val') -> Dict:
    """Evaluate the model"""
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'{phase}')
        for batch in pbar:
            # Move data to device
            fmri = batch['fmri'].to(device)
            smri = batch['smri'].to(device)
            labels = batch['label'].to(device)
            sites = batch['site'].to(device)
            ages = batch['age'].to(device)
            genders = batch['gender'].to(device)
            fiqs = batch['fiq'].to(device)
            
            # Forward pass
            class_logits, site_logits, age_pred, _ = model(
                fmri, smri, sites, ages, genders, fiqs
            )
            
            # Compute loss
            loss, _ = criterion(
                class_logits, site_logits, age_pred, 
                labels, sites, ages, model
            )
            
            total_loss += loss.item()
            
            # Get predictions
            probs = torch.softmax(class_logits, dim=1)
            preds = torch.argmax(class_logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    metrics = {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'auc': auc,
        'f1': f1,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }
    
    return metrics


# ============================================================================
# Main Training Function
# ============================================================================

def train_model(config: Dict, logger: logging.Logger):
    """
    Main training function
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set random seed
    set_seed(config['random_seed'])
    
    # Load data
    logger.info("Loading data...")

    # Load phenotypic data
    pheno_data = load_phenotypic_data(
        os.path.join(config['data_dir'], 'phynotypic'),
        config['num_samples'],
        logger
    )

    # Load fMRI data
    fmri_data = load_fmri_data(
        os.path.join(config['data_dir'], 'fMRI', 'CC200'),
        pheno_data['subject_IDs'],
        config['num_nodes'],
        logger
    )

    # Load sMRI data
    smri_data = load_smri_data(
        os.path.join(config['data_dir'], 'sMRI', 'freesurfer_stats'),
        pheno_data['subject_IDs'],
        logger
    )

    # Update config with actual data dimensions
    config['smri_dim'] = smri_data.shape[1]

    # Create site mapping
    unique_sites = np.unique(pheno_data['sites'])
    site_to_idx = {site: idx for idx, site in enumerate(unique_sites)}
    config['num_sites'] = len(unique_sites)
    
    # Create cross-validation splits
    fold_splits = create_site_aware_splits(
        pheno_data['sites'], 
        pheno_data['labels'],
        k_fold=config['k_fold'],
        random_state=config['random_seed'],
        logger=logger
    )
    
    # Train each fold
    fold_results = []
    
    for fold in range(config['k_fold']):
        logger.info(f"\n{'='*50}")
        logger.info(f"Training Fold {fold+1}/{config['k_fold']}")
        logger.info(f"{'='*50}")
        
        # Create datasets
        train_dataset = ABIDEDataset(
            fmri_data, smri_data, pheno_data['labels'],
            pheno_data['sites'], pheno_data['ages'], 
            pheno_data['genders'], pheno_data['fiqs'],
            fold_splits[fold]['train'], site_to_idx, augment=True
        )
        
        val_dataset = ABIDEDataset(
            fmri_data, smri_data, pheno_data['labels'],
            pheno_data['sites'], pheno_data['ages'], 
            pheno_data['genders'], pheno_data['fiqs'],
            fold_splits[fold]['val'], site_to_idx, augment=False
        )
        
        test_dataset = ABIDEDataset(
            fmri_data, smri_data, pheno_data['labels'],
            pheno_data['sites'], pheno_data['ages'], 
            pheno_data['genders'], pheno_data['fiqs'],
            fold_splits[fold]['test'], site_to_idx, augment=False
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                                 shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                               shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                                shuffle=False, num_workers=0)
        
        # Create model
        model = create_model(config).to(device)
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], 
                               weight_decay=config['weight_decay'])
        
        # Warmup + Cosine Annealing
        warmup_epochs = 10
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch) / float(max(1, warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (config['epochs'] - warmup_epochs)))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Create loss function
        criterion = MultiTaskLoss(
            lambda_cls=config['lambda_cls'],
            lambda_site=config['lambda_site'],
            lambda_age=config['lambda_age'],
            lambda_reg=config['lambda_reg']
        )
        
        # Training loop
        best_val_auc = 0
        patience_counter = 0
        
        for epoch in range(1, config['epochs'] + 1):
            # Train
            train_metrics = train_epoch(model, train_loader, criterion, 
                                       optimizer, device, epoch)
            
            # Validate
            val_metrics = evaluate(model, val_loader, criterion, device, 'Val')
            
            # Update scheduler
            scheduler.step()
            
            # Log metrics
            logger.info(f"Epoch {epoch}: "
                       f"Train Loss={train_metrics['loss']:.4f}, "
                       f"Train Acc={train_metrics['accuracy']:.4f}, "
                       f"Val Loss={val_metrics['loss']:.4f}, "
                       f"Val Acc={val_metrics['accuracy']:.4f}, "
                       f"Val AUC={val_metrics['auc']:.4f}")
            
            # Save best model
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                patience_counter = 0
                torch.save(model.state_dict(), 
                          os.path.join(config['save_dir'], f'best_model_fold{fold+1}.pth'))
                logger.info(f"Saved best model with Val AUC={best_val_auc:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= config['patience']:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model and evaluate on test set
        best_model_path = os.path.join(config['save_dir'], f'best_model_fold{fold+1}.pth')
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            test_metrics = evaluate(model, test_loader, criterion, device, 'Test')
        else:
            logger.warning(f"No best model found for fold {fold+1}, using current model state")
            test_metrics = evaluate(model, test_loader, criterion, device, 'Test')
        
        logger.info(f"\nFold {fold+1} Test Results:")
        logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"AUC: {test_metrics['auc']:.4f}")
        logger.info(f"F1: {test_metrics['f1']:.4f}")
        
        fold_results.append({
            'fold': fold + 1,
            'test_accuracy': test_metrics['accuracy'],
            'test_auc': test_metrics['auc'],
            'test_f1': test_metrics['f1']
        })
    
    # Aggregate results
    logger.info(f"\n{'='*50}")
    logger.info("Final Results Across All Folds")
    logger.info(f"{'='*50}")
    
    avg_accuracy = np.mean([r['test_accuracy'] for r in fold_results])
    avg_auc = np.mean([r['test_auc'] for r in fold_results])
    avg_f1 = np.mean([r['test_f1'] for r in fold_results])
    
    logger.info(f"Average Test Accuracy: {avg_accuracy:.4f} ± {np.std([r['test_accuracy'] for r in fold_results]):.4f}")
    logger.info(f"Average Test AUC: {avg_auc:.4f} ± {np.std([r['test_auc'] for r in fold_results]):.4f}")
    logger.info(f"Average Test F1: {avg_f1:.4f} ± {np.std([r['test_f1'] for r in fold_results]):.4f}")
    
    # Save results
    results = {
        'config': config,
        'fold_results': fold_results,
        'average_metrics': {
            'accuracy': avg_accuracy,
            'auc': avg_auc,
            'f1': avg_f1
        }
    }
    
    with open(os.path.join(config['save_dir'], 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    return results


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Configuration
    config = {
        # Data parameters
        'num_samples': 871,
        'num_nodes': 200,
        'smri_dim': 2500,
        'num_sites': 20,
        
        # Model parameters
        'hidden_dim': 256,
        'dropout': 0.3,
        
        # Training parameters
        'batch_size': 32,
        'learning_rate': 5e-4,  # Slightly higher with warmup
        'weight_decay': 0.01,  # Standard weight decay
        'epochs': 200,
        'patience': 30,
        'lambda_cls': 5.0,  # Increased weight for classification task
        'k_fold': 5,
        'random_seed': 42,
        
        # Loss weights
        'lambda_cls': 1.0,
        'lambda_site': 0.1,
        'lambda_age': 0.05,
        'lambda_reg': 0.001,
        
        # Paths
        'save_dir': './results_GNN',
        'data_dir': './data'
    }
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Setup logging
    logger = setup_logging(config['save_dir'])
    
    # Log configuration
    logger.info("Training Configuration:")
    logger.info(json.dumps(config, indent=4))
    
    # Train model
    results = train_model(config, logger)
    
    logger.info("\nTraining completed successfully!")
