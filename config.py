"""
Configuration file for CatBoost fMRI/sMRI classification
"""
import os

class Config:
    # Paths
    BASE_DIR = "/root/Multmodal_ABIDE/data"
    FMRI_ATLAS = 'CC200'  # AAL or CC200
    DATASET_PATH = "/root/Multmodal_ABIDE/data/fMRI/CC200/"
    LABEL_DIR = os.path.join(BASE_DIR, "phynotypic/")
    SMRI_DIR = os.path.join(BASE_DIR, "sMRI/freesurfer_stats/")
    
    # ComBat settings
    COMBAT_FMRI = True
    COMBAT_SMRI = True
    
    # Model save paths
    SAVE_PATH = os.path.join(BASE_DIR, f'save_models/{FMRI_ATLAS}_sMRI/')
    if COMBAT_SMRI:
        SAVE_PATH = os.path.join(SAVE_PATH, 'with_ComBat/')
    else:
        SAVE_PATH = os.path.join(SAVE_PATH, 'without_ComBat/')
    
    # Data parameters
    K_FOLD = 5
    NUM_SAMPLES = 871
    USELESS_SAMPLES = ['51334']
    
    # Feature selection parameters
    NEW_FEATURES_FMRI = 5000
    NEW_FEATURES_SMRI = 1435
    NEW_FEATURES_COMBINE = 6000
    
    # Image size based on atlas
    IMAGE_SIZE = [200, 200] if FMRI_ATLAS == 'CC200' else [116, 116]
    
    # Scaling
    USE_SCALER = True
    
    # CatBoost hyperparameters
    LEARNING_RATES = [0.001, 0.01, 0.05, 0.1, 0.2]
    DEPTHS = [4, 6, 8, 10]
    ITERATIONS = 100
    
    # sMRI feature parameters
    SMRI_ROI_DESIKAN = 68
    SMRI_ROI_ASEG = 45
    SMRI_ROI_WMPARC = 70
    
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
