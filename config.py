"""
Configuration file for CatBoost fMRI/sMRI classification
"""
import os

class Config:
    # Paths
    BASE_DIR = "/root/Multimoda-braingnn/"
    FMRI_ATLAS = 'CC200'  # AAL or CC200
    DATASET_PATH = "/root/Multimoda-braingnn/data/fMRI/CC200/"
    LABEL_DIR = os.path.join(BASE_DIR, "data/phynotypic/")
    SMRI_DIR = os.path.join(BASE_DIR, "data/sMRI/freesurfer_stats/")
    
    # ComBat harmonization settings
    # Corrects for site effects in neuroimaging data
    COMBAT_FMRI = True
    COMBAT_SMRI = True
    
    # Model save paths
    SAVE_PATH = os.path.join(BASE_DIR, f'results/save_models/{FMRI_ATLAS}_sMRI/')
    if COMBAT_SMRI:
        SAVE_PATH = os.path.join(SAVE_PATH, 'with_ComBat/')
    else:
        SAVE_PATH = os.path.join(SAVE_PATH, 'without_ComBat/')
    
    # Data parameters
    K_FOLD = 5
    NUM_SAMPLES = 871
    USELESS_SAMPLES = ['51334']  # Samples with missing or corrupted data
    
    # Feature selection parameters
    # fMRI: CC200 atlas = 200x200 connectivity matrix = 19,900 features (upper triangle)
    # Select ~25% to reduce dimensionality while preserving discriminative information
    NEW_FEATURES_FMRI = 5000
    
    # sMRI: 68 Desikan + 45 ASEG + 70 WMPARC = 183 ROIs × 7 features = 1281 features
    # Add phenotypic features (repeated 3x) ≈ 1435 total
    NEW_FEATURES_SMRI = 1435
    
    # Combined: ~30% reduction on concatenated features
    # Balances model complexity and information retention
    NEW_FEATURES_COMBINE = 6000
    
    # Image size based on atlas
    # CC200 = 200x200, AAL = 116x116
    IMAGE_SIZE = [200, 200] if FMRI_ATLAS == 'CC200' else [116, 116]
    
    # Feature scaling settings
    # Note: CatBoost doesn't require scaling, but we keep it for compatibility
    # with potential future models (e.g., neural networks)
    USE_SCALER = True
    
    # RFE (Recursive Feature Elimination) step sizes
    # fMRI: Aggressive elimination (step=100) due to high-dimensional features
    # sMRI: Conservative elimination (step=10) for more stable feature selection
    RFE_STEP_FMRI = 100
    RFE_STEP_SMRI = 10
    
    # CatBoost hyperparameters
    LEARNING_RATES = [0.001, 0.01, 0.05, 0.1, 0.2]
    DEPTHS = [4, 6, 8, 10]
    ITERATIONS = 100
    
    # sMRI feature parameters (number of ROIs per atlas)
    SMRI_ROI_DESIKAN = 68   # Desikan-Killiany cortical atlas
    SMRI_ROI_ASEG = 45      # ASEG subcortical atlas
    SMRI_ROI_WMPARC = 70    # White matter parcellation
    
    # Random seed for reproducibility
    RANDOM_SEED = 0
    
    # Visualization settings
    FIGURE_DPI = 300
    FIGURE_SIZE = (10, 8)
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        os.makedirs(cls.SAVE_PATH, exist_ok=True)
        os.makedirs(os.path.join(cls.SAVE_PATH, 'Max_voting/CatBoost/'), exist_ok=True)
        os.makedirs(os.path.join(cls.SAVE_PATH, 'figures/'), exist_ok=True)
