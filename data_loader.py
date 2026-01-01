"""
Data loading and preprocessing module
"""
import os
import numpy as np
import pandas as pd
import scipy.io as scio
from neuroCombat import neuroCombat
from config import Config


def get_ids(num_subjects=None, dir_path=''):
    """Load subject IDs from file"""
    subject_IDs = np.genfromtxt(os.path.join(dir_path, 'subject_IDs.txt'), dtype=str)
    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]
    return subject_IDs


def get_index(lst, item):
    """Get indices of item in list"""
    return [i for i in range(len(lst)) if lst[i] == item]


def load_fmri_data(config):
    """Load and process fMRI data"""
    print("Loading fMRI data...")
    
    # Load labels and metadata
    label = scio.loadmat(os.path.join(config.LABEL_DIR, 'ABIDE_label_871.mat'))['label'][0]
    site = scio.loadmat(os.path.join(config.LABEL_DIR, 'sites.mat'))['sites']
    gender = scio.loadmat(os.path.join(config.LABEL_DIR, 'genders.mat'))['genders']
    age = scio.loadmat(os.path.join(config.LABEL_DIR, 'ages.mat'))['ages']
    
    # Clean site names
    all_sites = [s.replace(' ', '') for s in site]
    unique_sites = np.unique(all_sites)
    
    # Initialize arrays
    num_valid = config.NUM_SAMPLES - len(config.USELESS_SAMPLES)
    length = config.IMAGE_SIZE[0] * (config.IMAGE_SIZE[1] - 1) // 2
    fMRI_images = np.zeros((num_valid, int(length)))
    labels = np.zeros(num_valid)
    sites = []
    genders_clean = np.zeros((num_valid, 1))
    ages_clean = np.zeros((num_valid, 1))
    
    # Load subject IDs
    subject_IDs = get_ids(config.NUM_SAMPLES, dir_path=config.LABEL_DIR).tolist()
    
    # Process each subject
    position = 0
    for i in range(config.NUM_SAMPLES):
        subject_name = subject_IDs[i]
        if subject_name in config.USELESS_SAMPLES:
            print(f'Skipping sample {subject_name}')
            continue
        
        # Load connectivity matrix
        image_name = f'{subject_name}.mat'
        image = scio.loadmat(os.path.join(config.DATASET_PATH, image_name))
        img = image['connectivity']
        
        # Extract upper triangle
        idx = np.triu_indices_from(img, 1)
        fMRI_images[position, :] = img[idx]
        
        # Store metadata
        subject_index = get_index(subject_IDs, subject_name)[0]
        sites.append(all_sites[subject_index])
        labels[position] = label[subject_index]
        genders_clean[position] = int(gender[subject_index]) + 1
        ages_clean[position] = float(age[subject_index].replace(' ', ''))
        position += 1
    
    return fMRI_images, labels, sites, genders_clean, ages_clean, unique_sites


def load_smri_data(config, subject_IDs):
    """Load and process sMRI data"""
    print("Loading sMRI data...")
    
    # Desikan-Killiany features
    dk_features = load_desikan_killiany(config, subject_IDs)
    
    # ASEG features
    aseg_features = load_aseg(config, subject_IDs)
    
    # WMPARC features
    wmparc_features = load_wmparc(config, subject_IDs)
    
    # Concatenate all sMRI features
    smri_features = np.concatenate((dk_features, aseg_features, wmparc_features), axis=1)
    
    return smri_features


def load_desikan_killiany(config, subject_IDs):
    """Load Desikan-Killiany atlas features"""
    print("  Loading Desikan-Killiany features...")
    
    feature_list = ['NumVert', 'SurfArea', 'GrayVol', 'ThickAvg', 
                   'ThickStd', 'MeanCurv', 'GausCurv', 'FoldInd', 'CurvInd']
    names = ['StructName'] + feature_list
    skiprows = list(range(0, 61))
    num_roi = config.SMRI_ROI_DESIKAN
    num_valid = config.NUM_SAMPLES - len(config.USELESS_SAMPLES)
    
    features = np.zeros((num_valid, num_roi * len(feature_list)))
    position = 0
    
    for i in range(config.NUM_SAMPLES):
        subject_name = subject_IDs[i]
        if subject_name in config.USELESS_SAMPLES:
            continue
        
        # Load left and right hemisphere
        table_left = pd.read_table(
            os.path.join(config.SMRI_DIR, subject_name, 'lh.aparc.stats'),
            sep='\\s+', names=names, skiprows=skiprows
        )
        table_right = pd.read_table(
            os.path.join(config.SMRI_DIR, subject_name, 'rh.aparc.stats'),
            sep='\\s+', names=names, skiprows=skiprows
        )
        
        for j, feature_name in enumerate(feature_list):
            left_feat = table_left[feature_name].values.tolist()
            right_feat = table_right[feature_name].values.tolist()
            all_feat = left_feat + right_feat
            features[position, j*num_roi:(j+1)*num_roi] = all_feat
        
        position += 1
    
    return features


def load_aseg(config, subject_IDs):
    """Load ASEG features"""
    print("  Loading ASEG features...")

    feature_list = ['Number of Voxels', 'Volume', 'Intensity normMean',
                   'Itensity normStdDev', 'Intensity normMin',
                   'Intensity normMax', 'Intensity normRange']
    names = ['ColHeader Index', 'Segmentation Id '] + ['Number of Voxels', 'Volume', 'Structure Name'] + feature_list[2:]
    skiprows = list(range(0, 79))
    num_roi = config.SMRI_ROI_ASEG
    num_valid = config.NUM_SAMPLES - len(config.USELESS_SAMPLES)

    features = np.zeros((num_valid, num_roi * len(feature_list)))
    position = 0

    for i in range(config.NUM_SAMPLES):
        subject_name = subject_IDs[i]
        if subject_name in config.USELESS_SAMPLES:
            continue

        table = pd.read_table(
            os.path.join(config.SMRI_DIR, subject_name, 'aseg.stats'),
            sep='\\s+', names=names, skiprows=skiprows
        )

        for j, feature_name in enumerate(feature_list):
            feat = table[feature_name].values.tolist()
            features[position, j*num_roi:(j+1)*num_roi] = feat

        position += 1

    return features


def load_wmparc(config, subject_IDs):
    """Load WMPARC features"""
    print("  Loading WMPARC features...")

    feature_list = ['Number of Voxels', 'Volume', 'Intensity normMean',
                   'Itensity normStdDev', 'Intensity normMin',
                   'Intensity normMax', 'Intensity normRange']
    names = ['ColHeader Index', 'Segmentation Id '] + ['Number of Voxels', 'Volume', 'Structure Name'] + feature_list[2:]
    skiprows = list(range(0, 65))
    num_roi = config.SMRI_ROI_WMPARC
    num_valid = config.NUM_SAMPLES - len(config.USELESS_SAMPLES)

    features = np.zeros((num_valid, num_roi * len(feature_list)))
    position = 0

    for i in range(config.NUM_SAMPLES):
        subject_name = subject_IDs[i]
        if subject_name in config.USELESS_SAMPLES:
            continue

        table = pd.read_table(
            os.path.join(config.SMRI_DIR, subject_name, 'wmparc.stats'),
            sep='\\s+', names=names, skiprows=skiprows
        )

        for j, feature_name in enumerate(feature_list):
            feat = table[feature_name].values.tolist()
            features[position, j*num_roi:(j+1)*num_roi] = feat

        position += 1

    return features


def load_phenotypic_data(config, subject_IDs):
    """Load phenotypic data (FIQ, NUM, PEC, RAT)"""
    print("Loading phenotypic data...")
    
    num_valid = config.NUM_SAMPLES - len(config.USELESS_SAMPLES)
    FIQ = np.zeros((num_valid, 1))
    NUM = np.zeros((num_valid, 1))
    PEC = np.zeros((num_valid, 1))
    RAT = np.zeros((num_valid, 1))
    
    # Load data
    FIQS = scio.loadmat(os.path.join(config.LABEL_DIR, 'FIQS.mat'))['FIQS']
    NUMS = scio.loadmat(os.path.join(config.LABEL_DIR, 'NUM.mat'))['NUM']
    PECS = scio.loadmat(os.path.join(config.LABEL_DIR, 'PEC.mat'))['PEC']
    RATS = scio.loadmat(os.path.join(config.LABEL_DIR, 'RAT.mat'))['RAT']
    
    position = 0
    for i in range(config.NUM_SAMPLES):
        subject_name = subject_IDs[i]
        if subject_name in config.USELESS_SAMPLES:
            continue
        
        subject_index = get_index(subject_IDs, subject_name)[0]
        FIQ[position] = int(FIQS[subject_index])
        NUM[position] = float(NUMS[subject_index])
        PEC[position] = float(PECS[subject_index])
        RAT[position] = int(RATS[subject_index])
        position += 1
    
    # Handle missing FIQ values
    FIQ[FIQ == -9999] = 108
    
    return FIQ, NUM, PEC, RAT


def apply_combat(data, sites, labels, genders, ages, unique_sites, combat_flag):
    """Apply ComBat harmonization"""
    if not combat_flag:
        return data
    
    print("Applying ComBat harmonization...")
    
    # Prepare batch labels
    batch = [get_index(unique_sites.tolist(), site)[0] + 1 for site in sites]
    
    # Prepare covariates
    covars = pd.DataFrame({
        'batch': batch,
        'labels': [int(l) + 1 for l in labels],
        'genders': genders.flatten(),
        'ages': ages.flatten()
    })
    
    # Apply neuroCombat
    data_harmonized = neuroCombat(
        dat=data.T,
        covars=covars,
        batch_col='batch',
        categorical_cols=['labels', 'genders'],
        continuous_cols=['ages']
    )["data"]
    
    return data_harmonized.T
