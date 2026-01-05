"""
Model training and evaluation utilities
"""
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
from sklearn import metrics
from catboost import CatBoostClassifier
from config import Config


def feature_selection_fmri(matrix, labels, train_ind, fnum, scaler_flag=True):
    """
    Feature selection for fMRI data using RFE
    
    Fits scaler and RFE selector ONLY on training data to prevent data leakage.
    
    Args:
        matrix: feature matrix (num_subjects x num_features)
        labels: ground truth labels (num_subjects x 1)
        train_ind: indices of training samples
        fnum: number of features to select
        scaler_flag: whether to apply StandardScaler
    
    Returns:
        selector: fitted RFE selector (trained only on training data)
        scaler: fitted StandardScaler (fitted only on training data) or None
    """
    # Fit scaler ONLY on training data
    scaler = None
    if scaler_flag:
        scaler = StandardScaler()
        train_data = matrix[train_ind, :]
        scaler.fit(train_data)
        X_train_scaled = scaler.transform(train_data)
    else:
        X_train_scaled = matrix[train_ind, :]
    
    # Fit selector ONLY on training data
    estimator = RidgeClassifier()
    selector = RFE(estimator, n_features_to_select=fnum, step=100, verbose=0)
    y_train = labels[train_ind]
    selector.fit(X_train_scaled, y_train.ravel())
    
    return selector, scaler


def feature_selection_smri(matrix, labels, train_ind, fnum, scaler_flag=True):
    """
    Feature selection for sMRI data using RFE
    
    Conservative step size (10) for more gradual feature elimination.
    Fits scaler and RFE selector ONLY on training data to prevent data leakage.
    
    Args:
        matrix: feature matrix (num_subjects x num_features)
        labels: ground truth labels (num_subjects x 1)
        train_ind: indices of training samples
        fnum: number of features to select
        scaler_flag: whether to apply StandardScaler
    
    Returns:
        selector: fitted RFE selector (trained only on training data)
        scaler: fitted StandardScaler (fitted only on training data) or None
    """
    # Fit scaler ONLY on training data
    scaler = None
    if scaler_flag:
        scaler = StandardScaler()
        train_data = matrix[train_ind, :]
        scaler.fit(train_data)
        X_train_scaled = scaler.transform(train_data)
    else:
        X_train_scaled = matrix[train_ind, :]
    
    # Fit selector ONLY on training data
    estimator = RidgeClassifier()
    selector = RFE(estimator, n_features_to_select=fnum, step=10, verbose=0)
    y_train = labels[train_ind]
    selector.fit(X_train_scaled, y_train.ravel())
    
    return selector, scaler


def train_catboost(train_data, train_labels, val_data, val_labels, 
                   learning_rates, depths, config, verbose=0):
    """
    Train CatBoost with hyperparameter search
    
    Automatically selects GPU if available, falls back to CPU.
    
    Returns:
        best_model: trained model with best validation accuracy
        best_lr: best learning rate
        best_depth: best depth
        val_acc: validation accuracy
    """
    # Auto-detect GPU availability
    try:
        import torch
        has_gpu = torch.cuda.is_available()
        device_type = 'GPU'
    except (ImportError, RuntimeError):
        has_gpu = False
        device_type = 'CPU'
    
    if has_gpu:
        print("GPU detected, using GPU for training")
        task_type = 'GPU'
        devices = '0'
    else:
        print("GPU not available, falling back to CPU")
        task_type = 'CPU'
        devices = None
    
    val_special_accuracy = 0
    best_model = None
    best_lr = None
    best_depth = None
    
    for lr in learning_rates:
        for depth in depths:
            model = CatBoostClassifier(
                iterations=config.ITERATIONS,
                learning_rate=lr,
                depth=depth,
                verbose=verbose,
                random_state=config.RANDOM_SEED,
                task_type=task_type,
                devices=devices
            )
            
            model.fit(train_data, train_labels)
            val_results = model.predict(val_data)
            val_accuracy = metrics.accuracy_score(val_labels, val_results)
            
            if val_accuracy > val_special_accuracy:
                val_special_accuracy = val_accuracy
                best_model = model
                best_lr = lr
                best_depth = depth
    
    return best_model, best_lr, best_depth, val_special_accuracy


def evaluate_model(model, test_data, test_labels):
    """Evaluate model on test data"""
    test_results = model.predict(test_data)
    test_results_prob = model.predict_proba(test_data)
    test_accuracy = metrics.accuracy_score(test_labels, test_results)
    
    return test_results, test_results_prob, test_accuracy


def save_model(model, path):
    """Save model to disk"""
    joblib.dump(model, path)


def load_model(path):
    """Load model from disk"""
    return joblib.load(path)
