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
    
    Args:
        matrix: feature matrix (num_subjects x num_features)
        labels: ground truth labels (num_subjects x 1)
        train_ind: indices of training samples
        fnum: number of features to select
        scaler_flag: whether to apply StandardScaler
    
    Returns:
        selector: fitted RFE selector
        scaler: fitted StandardScaler (or None)
        matrix_scaled: scaled feature matrix
    """
    scaler = None
    matrix_scaled = matrix.copy()
    
    if scaler_flag:
        scaler = StandardScaler()
        featureX = matrix[train_ind, :]
        featureX_scaled = scaler.fit_transform(featureX)
        matrix_scaled[train_ind, :] = featureX_scaled
    else:
        featureX_scaled = matrix[train_ind, :]
    
    estimator = RidgeClassifier()
    selector = RFE(estimator, n_features_to_select=fnum, step=100, verbose=0)
    
    featureY = labels[train_ind]
    selector = selector.fit(featureX_scaled, featureY.ravel())
    
    return selector, scaler, matrix_scaled


def feature_selection_smri(matrix, labels, train_ind, fnum, scaler_flag=True):
    """
    Feature selection for sMRI data using RFE
    Similar to fMRI but with different step size
    """
    scaler = None
    matrix_scaled = matrix.copy()
    
    if scaler_flag:
        scaler = StandardScaler()
        featureX = matrix[train_ind, :]
        featureX_scaled = scaler.fit_transform(featureX)
        matrix_scaled[train_ind, :] = featureX_scaled
    else:
        featureX_scaled = matrix[train_ind, :]
    
    estimator = RidgeClassifier()
    selector = RFE(estimator, n_features_to_select=fnum, step=10, verbose=0)
    
    featureY = labels[train_ind]
    selector = selector.fit(featureX_scaled, featureY.ravel())
    
    return selector, scaler, matrix_scaled


def train_catboost(train_data, train_labels, val_data, val_labels, 
                   learning_rates, depths, config, verbose=0):
    """
    Train CatBoost with hyperparameter search
    
    Returns:
        best_model: trained model with best validation accuracy
        best_lr: best learning rate
        best_depth: best depth
        val_acc: validation accuracy
    """
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
                task_type='GPU',
                devices='0'
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
