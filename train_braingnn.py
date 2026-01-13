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
import re

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
    plot_roc_curve,
    plot_confusion_matrix,
    plot_training_history,
    plot_roc_per_fold,
    plot_site_performance,
    plot_clinical_metrics,
    print_classification_report,
    save_results_to_csv
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
                 # fiq removed
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
        # Map site names to indices
        self.sites = np.array([site_to_idx[sites[i]] for i in indices])

        # Normalize continuous phenotypic variables (global z-score)
        # Compute global statistics from the full arrays (not just indices)
        try:
            age_mean = float(np.nanmean(ages))
            age_std = float(np.nanstd(ages)) + 1e-8
        except:
            age_mean, age_std = 0.0, 1.0

        self.ages = (ages[indices] - age_mean) / age_std
        self.genders = genders[indices]
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
            'gender': torch.LongTensor([self.genders[idx]])[0]
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
    
    # Robust parser: extract all floating-point numbers from relevant stats files
    def parse_stats_file(file_path: str) -> List[float]:
        nums: List[float] = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as fh:
                for line in fh:
                    # Find all numbers in the line (integers and floats)
                    found = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", line)
                    for v in found:
                        try:
                            nums.append(float(v))
                        except:
                            continue
        except Exception as e:
            logger.debug(f"Failed to parse {file_path}: {e}")
        return nums

    all_features: List[List[float]] = []

    for i, subject_id in enumerate(tqdm(subject_IDs, desc="Loading sMRI")):
        subject_features: List[float] = []
        subject_dir = os.path.join(freesurfer_path, subject_id)

        # Try common files
        for fname in ['lh.aparc.stats', 'rh.aparc.stats', 'aseg.stats', 'lh.aparc.pial.stats', 'rh.aparc.pial.stats']:
            file_path = os.path.join(subject_dir, fname)
            if os.path.exists(file_path):
                parsed = parse_stats_file(file_path)
                if parsed:
                    subject_features.extend(parsed)

        if len(subject_features) == 0:
            logger.debug(f"No sMRI features parsed for {subject_id}; leaving empty vector")

        all_features.append(subject_features)

    # Convert to numpy array and handle variable lengths
    if len(all_features) == 0:
        smri_data = np.zeros((len(subject_IDs), 0))
    else:
        max_len = max((len(f) for f in all_features), default=0)
        if max_len == 0:
            smri_data = np.zeros((len(subject_IDs), 0))
        else:
            smri_data = np.zeros((len(subject_IDs), max_len))
            for i, features in enumerate(all_features):
                if len(features) > 0:
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
    
    # FIQ removed — do not load or return FIQ values
    
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
        'sites': sites,
        'subject_IDs': subject_IDs.tolist()
    }


# ============================================================================
# Cross-Validation Split Function
# ============================================================================

def create_stratified_splits(labels: np.ndarray, 
                             k_fold: int = 5, random_state: int = 42,
                             logger: Optional[logging.Logger] = None) -> Dict:
    """
    Create standard stratified k-fold splits (assuming site effects are removed)
    
    Returns:
        Dictionary with train, val, test indices for each fold
    """
    if logger:
        logger.info(f"Creating {k_fold}-fold standard stratified splits")
    
    num_samples = len(labels)
    indices = np.arange(num_samples)
    fold_splits = {}
    
    # Outer split for test set
    skf_outer = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=random_state)
    
    for fold, (train_val_idx, test_idx) in enumerate(skf_outer.split(indices, labels)):
        # Inner split for validation set
        train_val_labels = labels[train_val_idx]
        
        # Use 20% of train_val for validation
        skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        for inner_train_idx, inner_val_idx in skf_inner.split(train_val_idx, train_val_labels):
            train_idx = train_val_idx[inner_train_idx]
            val_idx = train_val_idx[inner_val_idx]
            break # Only need one split
            
        fold_splits[fold] = {
            'train': train_idx.tolist(),
            'val': val_idx.tolist(),
            'test': test_idx.tolist()
        }
        
        if logger:
            logger.info(f"Fold {fold+1}: Train={len(train_idx)}, "
                       f"Val={len(val_idx)}, Test={len(test_idx)}")
    
    return fold_splits


# ============================================================================
# Training Functions
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, label_smoothing=0.1, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        # Use no reduction so we can compute focal per-sample
        self.ce = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing, weight=weight)

    def forward(self, inputs, targets):
        # inputs: (batch, num_classes), targets: (batch,)
        ce_loss = self.ce(inputs, targets)  # per-sample
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        # return mean over batch
        return focal_loss.mean()

class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with weighted components
    """
    def __init__(self, lambda_cls: float = 5.0, lambda_site: float = 0.1, 
                 lambda_age: float = 0.05, lambda_reg: float = 0.001,
                 class_weights: Optional[torch.Tensor] = None):
        super(MultiTaskLoss, self).__init__()
        self.lambda_cls = lambda_cls
        self.lambda_site = lambda_site
        self.lambda_age = lambda_age
        self.lambda_reg = lambda_reg
        
        self.cls_criterion = FocalLoss(label_smoothing=0.1, weight=class_weights)
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
        # Forward pass (FIQ removed)
        class_logits, site_logits, age_pred, _ = model(
            fmri, smri, sites, ages, genders
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
    all_sites = []
    
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
            # Forward pass (FIQ removed)
            class_logits, site_logits, age_pred, _ = model(
                fmri, smri, sites, ages, genders
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
            all_sites.extend(sites.cpu().numpy())
    
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
        'labels': all_labels,
        'sites': all_sites,
        'test_labels': all_labels,
        'test_probs': np.column_stack([1 - np.array(all_probs), np.array(all_probs)])
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

    # -----------------
    # Normalization
    # -----------------
    # fMRI: assume connectivity (correlation) matrices — apply Fisher z and per-subject z-score
    try:
        fmri_clipped = np.clip(fmri_data, -0.999, 0.999)
        fmri_data = np.arctanh(fmri_clipped)  # Fisher z
        # standardize per-subject
        fmri_mean = fmri_data.mean(axis=(1,2), keepdims=True)
        fmri_std = fmri_data.std(axis=(1,2), keepdims=True) + 1e-8
        fmri_data = (fmri_data - fmri_mean) / fmri_std
        logger.info('Applied Fisher z and per-subject z-score to fMRI data')
    except Exception as e:
        logger.warning(f'Failed to normalize fMRI data: {e}')

    # Global sMRI normalization removed to prevent data leakage.
    # Normalization will be performed per-fold using training set statistics.
    logger.info('Global sMRI normalization skipped (will be performed per-fold)')

    # Update config with actual data dimensions
    config['smri_dim'] = smri_data.shape[1]

    # Create site mapping
    unique_sites = np.unique(pheno_data['sites'])
    site_to_idx = {site: idx for idx, site in enumerate(unique_sites)}
    config['num_sites'] = len(unique_sites)
    
    # Create cross-validation splits
    fold_splits = create_stratified_splits(
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
            pheno_data['genders'],
            fold_splits[fold]['train'], site_to_idx, augment=True
        )
        
        val_dataset = ABIDEDataset(
            fmri_data, smri_data, pheno_data['labels'],
            pheno_data['sites'], pheno_data['ages'], 
            pheno_data['genders'],
            fold_splits[fold]['val'], site_to_idx, augment=False
        )
        
        test_dataset = ABIDEDataset(
            fmri_data, smri_data, pheno_data['labels'],
            pheno_data['sites'], pheno_data['ages'], 
            pheno_data['genders'],
            fold_splits[fold]['test'], site_to_idx, augment=False
        )

        # Per-fold sMRI normalization to prevent data leakage
        if smri_data.shape[1] > 0:
            scaler = StandardScaler()
            # Fit on training data only
            train_dataset.smri_data = scaler.fit_transform(train_dataset.smri_data)
            # Transform val and test data using training statistics
            val_dataset.smri_data = scaler.transform(val_dataset.smri_data)
            test_dataset.smri_data = scaler.transform(test_dataset.smri_data)
            
            # Handle potential NaNs from constant features
            train_dataset.smri_data = np.nan_to_num(train_dataset.smri_data)
            val_dataset.smri_data = np.nan_to_num(val_dataset.smri_data)
            test_dataset.smri_data = np.nan_to_num(test_dataset.smri_data)
            logger.info(f"Fold {fold+1}: Applied per-fold sMRI normalization")
        
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
        
        # Calculate class weights for this fold
        train_labels_fold = pheno_data['labels'][fold_splits[fold]['train']]
        class_counts = np.bincount(train_labels_fold.astype(int))
        class_weights = torch.tensor([len(train_labels_fold) / (len(class_counts) * count) for count in class_counts], dtype=torch.float32).to(device)
        
        # Create loss function
        criterion = MultiTaskLoss(
            lambda_cls=config['lambda_cls'],
            lambda_site=config['lambda_site'],
            lambda_age=config['lambda_age'],
            lambda_reg=config['lambda_reg'],
            class_weights=class_weights
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
        
        # =====================================================================
        # Visualization for this fold
        # =====================================================================
        fold_vis_dir = os.path.join(config['save_dir'], f'fold_{fold+1}')
        os.makedirs(fold_vis_dir, exist_ok=True)
        
        # Convert predictions to numpy array for visualization
        test_preds_array = np.array(test_metrics['predictions'])
        test_labels_array = np.array(test_metrics['labels'])
        test_probs_array = test_metrics['test_probs']
        test_sites_array = np.array(test_metrics['sites'])
        
        # Plot ROC curve
        try:
            roc_auc = plot_roc_curve(
                test_labels_array, 
                test_probs_array,
                save_path=os.path.join(fold_vis_dir, 'roc_curve.png'),
                title=f'ROC Curve - Fold {fold+1}'
            )
            logger.info(f"Saved ROC curve plot for Fold {fold+1}")
        except Exception as e:
            logger.warning(f"Failed to plot ROC curve for Fold {fold+1}: {e}")
        
        # Plot confusion matrix
        try:
            plot_confusion_matrix(
                test_labels_array,
                test_preds_array,
                save_path=os.path.join(fold_vis_dir, 'confusion_matrix.png'),
                title=f'Confusion Matrix - Fold {fold+1}'
            )
            logger.info(f"Saved confusion matrix plot for Fold {fold+1}")
        except Exception as e:
            logger.warning(f"Failed to plot confusion matrix for Fold {fold+1}: {e}")
        
        # Print classification report
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Classification Report - Fold {fold+1}")
            logger.info(f"{'='*60}")
            print_classification_report(test_labels_array, test_preds_array)
        except Exception as e:
            logger.warning(f"Failed to print classification report for Fold {fold+1}: {e}")
        
        # Plot clinical metrics
        try:
            plot_clinical_metrics(
                test_labels_array,
                test_preds_array,
                test_probs_array,
                save_path=os.path.join(fold_vis_dir, 'clinical_metrics.png')
            )
            logger.info(f"Saved clinical metrics plot for Fold {fold+1}")
        except Exception as e:
            logger.warning(f"Failed to plot clinical metrics for Fold {fold+1}: {e}")
        
        # Create site mapping for visualization
        idx_to_site = {v: k for k, v in site_to_idx.items()}
        test_sites_names = np.array([idx_to_site[s] for s in test_sites_array])
        
        # Plot site performance
        try:
            plot_site_performance(
                test_sites_names,
                test_labels_array,
                test_preds_array,
                test_probs_array,
                save_path=os.path.join(fold_vis_dir, 'site_performance.png')
            )
            logger.info(f"Saved site performance plot for Fold {fold+1}")
        except Exception as e:
            logger.warning(f"Failed to plot site performance for Fold {fold+1}: {e}")
        
        # Save fold results to CSV
        try:
            fold_results_dict = {
                'sample_id': range(len(test_labels_array)),
                'true_label': test_labels_array,
                'predicted_label': test_preds_array,
                'prediction_probability': test_probs_array[:, 1],
                'site': test_sites_names
            }
            save_results_to_csv(
                fold_results_dict,
                os.path.join(fold_vis_dir, 'fold_results.csv')
            )
        except Exception as e:
            logger.warning(f"Failed to save fold results CSV for Fold {fold+1}: {e}")
        
        fold_results.append({
            'fold': fold + 1,
            'test_accuracy': test_metrics['accuracy'],
            'test_auc': test_metrics['auc'],
            'test_f1': test_metrics['f1'],
            'test_labels': test_labels_array,
            'test_preds': test_preds_array,
            'test_probs': test_probs_array,
            'test_sites': test_sites_array
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
    
    # =========================================================================
    # Cross-fold Visualization
    # =========================================================================
    logger.info(f"\n{'='*50}")
    logger.info("Generating Cross-Fold Visualizations")
    logger.info(f"{'='*50}")
    
    # Plot ROC curves across all folds
    try:
        fold_results_for_roc = [
            {
                'test_labels': r['test_labels'],
                'test_probs': r['test_probs']
            }
            for r in fold_results
        ]
        mean_auc = plot_roc_per_fold(
            fold_results_for_roc,
            save_path=os.path.join(config['save_dir'], 'roc_per_fold.png'),
            title='ROC Curves Per Fold'
        )
        logger.info(f"Saved ROC per fold plot (Mean AUC: {mean_auc:.4f})")
    except Exception as e:
        logger.warning(f"Failed to plot ROC per fold: {e}")
    
    # Plot training history
    try:
        history_dict = {
            'fold': [r['fold'] for r in fold_results],
            'val_acc': [r['test_accuracy'] for r in fold_results],  # Using test as val for this plot
            'test_acc': [r['test_accuracy'] for r in fold_results]
        }
        plot_training_history(
            history_dict,
            save_path=os.path.join(config['save_dir'], 'training_history.png')
        )
        logger.info("Saved training history plot")
    except Exception as e:
        logger.warning(f"Failed to plot training history: {e}")
    
    # Aggregate predictions from all folds for overall site performance
    try:
        all_test_labels = np.concatenate([r['test_labels'] for r in fold_results])
        all_test_preds = np.concatenate([r['test_preds'] for r in fold_results])
        all_test_probs = np.vstack([r['test_probs'] for r in fold_results])
        all_test_sites = np.concatenate([r['test_sites'] for r in fold_results])
        
        # Create site mapping for visualization
        idx_to_site = {v: k for k, v in site_to_idx.items()}
        all_test_sites_names = np.array([idx_to_site[s] for s in all_test_sites])
        
        # Plot overall site performance
        plot_site_performance(
            all_test_sites_names,
            all_test_labels,
            all_test_preds,
            all_test_probs,
            save_path=os.path.join(config['save_dir'], 'overall_site_performance.png')
        )
        logger.info("Saved overall site performance plot")
    except Exception as e:
        logger.warning(f"Failed to plot overall site performance: {e}")
    
    # Plot overall clinical metrics
    try:
        plot_clinical_metrics(
            all_test_labels,
            all_test_preds,
            all_test_probs,
            save_path=os.path.join(config['save_dir'], 'overall_clinical_metrics.png')
        )
        logger.info("Saved overall clinical metrics plot")
    except Exception as e:
        logger.warning(f"Failed to plot overall clinical metrics: {e}")
    
    # Print overall classification report
    try:
        logger.info(f"\n{'='*60}")
        logger.info("Overall Classification Report (All Folds)")
        logger.info(f"{'='*60}")
        print_classification_report(all_test_labels, all_test_preds)
    except Exception as e:
        logger.warning(f"Failed to print overall classification report: {e}")
    
    # Save overall results to CSV
    try:
        # Build fold mapping - track which fold each sample belongs to
        fold_mapping = []
        for r in fold_results:
            fold_idx = r['fold']
            fold_mapping.extend([fold_idx] * len(r['test_labels']))
        
        overall_results_dict = {
            'fold': fold_mapping,
            'true_label': all_test_labels.astype(int).tolist(),
            'predicted_label': all_test_preds.astype(int).tolist(),
            'prediction_probability': all_test_probs[:, 1].astype(float).tolist(),
            'site': all_test_sites_names.tolist()
        }
        save_results_to_csv(
            overall_results_dict,
            os.path.join(config['save_dir'], 'overall_results.csv')
        )
        logger.info("Saved overall results CSV")
    except Exception as e:
        logger.warning(f"Failed to save overall results CSV: {e}")
    
    # Save summary results to CSV
    try:
        summary_results = {
            'Metric': ['Accuracy', 'AUC', 'F1-Score'],
            'Mean': [avg_accuracy, avg_auc, avg_f1],
            'Std': [
                np.std([r['test_accuracy'] for r in fold_results]),
                np.std([r['test_auc'] for r in fold_results]),
                np.std([r['test_f1'] for r in fold_results])
            ]
        }
        save_results_to_csv(
            summary_results,
            os.path.join(config['save_dir'], 'summary_metrics.csv')
        )
        logger.info("Saved summary metrics CSV")
    except Exception as e:
        logger.warning(f"Failed to save summary metrics CSV: {e}")
    
    # Save results - only serialize metrics, not numpy arrays
    serializable_fold_results = [
        {
            'fold': r['fold'],
            'test_accuracy': float(r['test_accuracy']),
            'test_auc': float(r['test_auc']),
            'test_f1': float(r['test_f1'])
        }
        for r in fold_results
    ]
    
    results = {
        'config': config,
        'fold_results': serializable_fold_results,
        'average_metrics': {
            'accuracy': float(avg_accuracy),
            'auc': float(avg_auc),
            'f1': float(avg_f1)
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
        'hidden_dim': 128,      # Reduced to prevent overfitting
        'dropout': 0.4,         # Moderate dropout
        
        # Training parameters
        'batch_size': 32,
        'learning_rate': 5e-4,
        'weight_decay': 0.02,   # Increased weight decay for regularization
        'epochs': 200,
        'patience': 30,
        'k_fold': 5,
        'random_seed': 42,
        
        # Loss weights
        'lambda_cls': 5.0,      # Prioritize classification
        'lambda_site': 0.05,    # Domain adaptation weight
        'lambda_age': 0.01,     # Deconfounding weight
        'lambda_reg': 0.001,    # Stronger L2 regularization weight
        
        # Paths
        'save_dir': './results_test',
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