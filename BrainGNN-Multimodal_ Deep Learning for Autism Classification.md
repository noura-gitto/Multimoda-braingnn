# BrainGNN-Multimodal: Deep Learning for Autism Classification

A state-of-the-art deep learning framework for autism spectrum disorder (ASD) classification using multimodal neuroimaging data from the ABIDE dataset.

## 🎯 Key Features

- **Graph Neural Networks** for fMRI connectivity matrices
- **Deep Neural Networks** for sMRI morphometric features  
- **Multimodal Fusion** with cross-modal attention mechanisms
- **Domain Adaptation** for handling multi-site heterogeneity
- **Multi-task Learning** with auxiliary tasks (site prediction, age regression)
- **Interpretability** through attention visualization
- **Site-aware Cross-Validation** for robust evaluation

## 📊 Expected Performance

| Metric | SVM Baseline | BrainGNN-Multimodal | Improvement |
|--------|--------------|---------------------|-------------|
| Accuracy | 65-70% | **75-85%** | +8-15% |
| AUC-ROC | 0.70-0.75 | **0.80-0.90** | +0.10-0.15 |
| F1-Score | 0.65-0.70 | **0.75-0.85** | +0.10-0.15 |

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Data                               │
├─────────────────────────────────────────────────────────────┤
│  fMRI (200×200)  │  sMRI (2500)  │  Phenotypic (age/site)   │
└────────┬──────────┴───────┬───────┴──────────┬──────────────┘
         │                  │                   │
         ▼                  ▼                   ▼
┌─────────────────┐ ┌──────────────┐ ┌──────────────────┐
│  Graph Neural   │ │  Deep Neural │ │   Phenotypic     │
│    Network      │ │   Network    │ │    Embedding     │
│                 │ │              │ │                  │
│ • GCN Layers    │ │ • Attention  │ │ • Site Embed     │
│ • GAT Layer     │ │ • Residual   │ │ • Age/Gender     │
│ • Graph Pool    │ │ • Feature    │ │ • FIQ Encoding   │
│ • Self-Attn     │ │   Selection  │ │                  │
└────────┬────────┘ └──────┬───────┘ └────────┬─────────┘
         │                  │                   │
         └──────────────────┴───────────────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  Multimodal Fusion   │
                 │                      │
                 │ • Cross-Modal Attn   │
                 │ • Bilinear Pooling   │
                 │ • Feature Concat     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  Classification Head │
                 │                      │
                 │ • Main: ASD vs TD    │
                 │ • Aux: Site Pred     │
                 │ • Aux: Age Regress   │
                 └──────────────────────┘
```

## 📁 File Structure

```
.
├── braingnn_multimodal.py      # Model architecture
├── train_braingnn.py           # Training pipeline
├── README_DeepLearning.md      # This file
├── deep_learning_architecture.md  # Detailed architecture document
├── requirements.txt            # Python dependencies
└── results/                    # Training results and models
    ├── best_model_fold1.pth
    ├── best_model_fold2.pth
    ├── ...
    └── results.json
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install PyTorch (CPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install numpy scipy pandas scikit-learn tqdm matplotlib seaborn

# For GPU support (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Data Preparation

Your data should be organized as follows:

```
data/
├── fMRI/
│   ├── CC200/  or  AAL/
│   │   ├── 50001.mat
│   │   ├── 50002.mat
│   │   └── ...
├── sMRI/
│   └── freesurfer_stats/
│       ├── 50001/
│       │   ├── lh.aparc.stats
│       │   ├── rh.aparc.stats
│       │   ├── aseg.stats
│       │   └── ...
│       └── ...
└── phenotypic/
    ├── ABIDE_label_871.mat
    ├── ages.mat
    ├── genders.mat
    ├── FIQS.mat
    ├── sites.mat
    └── subject_IDs.txt
```

### 3. Training

**Basic Training:**
```bash
python train_braingnn.py
```

**Custom Configuration:**
```python
# Edit config in train_braingnn.py
config = {
    'num_nodes': 200,           # 200 for CC200, 116 for AAL
    'smri_dim': 2500,           # Adjust based on your features
    'batch_size': 32,
    'learning_rate': 1e-3,
    'epochs': 200,
    'k_fold': 5,
    # ... other parameters
}
```

### 4. Evaluation

Results will be saved in `results/results.json`:

```json
{
    "average_metrics": {
        "accuracy": 0.82,
        "auc": 0.87,
        "f1": 0.81
    },
    "fold_results": [
        {"fold": 1, "test_accuracy": 0.83, "test_auc": 0.88},
        ...
    ]
}
```

## 🔧 Model Components

### 1. fMRI Graph Neural Network Branch

Processes functional connectivity matrices as graphs:

```python
class fMRIGraphBranch(nn.Module):
    - Graph Convolutional Layers (3 layers)
    - Graph Attention Layer (GAT)
    - Graph Pooling (Top-K)
    - Self-Attention for global features
```

**Key Features:**
- Preserves brain network topology
- Learns from connectivity patterns
- Captures multi-scale features

### 2. sMRI Deep Neural Network Branch

Processes morphometric features:

```python
class sMRIBranch(nn.Module):
    - Feature Embedding
    - Multi-head Self-Attention
    - Residual Blocks (2 blocks)
    - Channel-wise Attention
```

**Key Features:**
- Automatic feature learning
- Attention-based feature selection
- Deep hierarchical representations

### 3. Phenotypic Embedding Branch

Encodes demographic information:

```python
class PhenotypicBranch(nn.Module):
    - Site Embedding (for domain adaptation)
    - Age Encoder
    - Gender Embedding
    - FIQ Encoder
```

**Key Features:**
- Handles categorical and continuous variables
- Enables domain adaptation
- Controls for confounders

### 4. Multimodal Fusion Layer

Combines information from all modalities:

```python
class MultimodalFusion(nn.Module):
    - Cross-Modal Attention (fMRI ↔ sMRI)
    - Bilinear Pooling (second-order interactions)
    - Feature Concatenation
```

**Key Features:**
- Learns complementary information
- Models inter-modal interactions
- Sophisticated fusion strategy

### 5. Classification Head

Multi-task learning for better generalization:

```python
class ClassificationHead(nn.Module):
    - Main Task: ASD vs TD classification
    - Auxiliary Task 1: Site prediction (domain adaptation)
    - Auxiliary Task 2: Age regression (deconfounding)
```

## 📈 Training Strategy

### Loss Function

```python
total_loss = (
    λ_cls * classification_loss +      # Main task (λ=1.0)
    λ_site * site_prediction_loss +    # Domain adaptation (λ=0.1)
    λ_age * age_regression_loss +      # Deconfounding (λ=0.05)
    λ_reg * L2_regularization          # Weight decay (λ=0.001)
)
```

### Data Augmentation

**fMRI:**
- Gaussian noise injection (σ=0.01)
- Random edge dropout (10-20%)
- Node feature masking

**sMRI:**
- Gaussian noise (σ=0.05)
- Feature dropout (10%)

### Regularization

- **Dropout**: 0.3-0.5 in fully connected layers
- **Batch Normalization**: After each linear layer
- **Weight Decay**: L2 penalty (0.01)
- **Early Stopping**: Patience of 20 epochs
- **Gradient Clipping**: Max norm = 1.0

### Optimization

- **Optimizer**: AdamW
- **Learning Rate**: 1e-3 with cosine annealing
- **Batch Size**: 32
- **Epochs**: 200 (with early stopping)

## 🎨 Visualization and Interpretation

### Attention Visualization

```python
# Get attention weights
_, _, _, attention_dict = model(fmri_data, smri_data, ...)

# Visualize fMRI attention
fmri_attention = attention_dict['fmri_attention']
# Plot attention heatmap

# Visualize sMRI feature attention
smri_attention = attention_dict['smri_attention']
# Plot feature importance
```

### Feature Embeddings

```python
# Extract learned embeddings
embeddings = model.get_embeddings(fmri_data, smri_data, ...)

# Visualize with t-SNE or UMAP
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
embeddings_2d = tsne.fit_transform(embeddings.cpu().numpy())
```

## 📊 Comparison with SVM

| Aspect | SVM (Baseline) | BrainGNN-Multimodal |
|--------|----------------|---------------------|
| **Feature Engineering** | Manual (RFE) | Automatic |
| **Multimodal Fusion** | Concatenation | Cross-modal attention |
| **Graph Structure** | Ignored | Explicitly modeled |
| **Domain Adaptation** | None | Site adversarial learning |
| **Interpretability** | Feature weights | Attention maps |
| **Training Time** | Hours (grid search) | Hours (GPU) |
| **Accuracy** | 65-70% | **75-85%** |
| **AUC** | 0.70-0.75 | **0.80-0.90** |

## 🔬 Advanced Features

### 1. Transfer Learning

Pre-train on larger datasets (UK Biobank, HCP):

```python
# Load pre-trained weights
pretrained_weights = torch.load('pretrained_model.pth')
model.load_state_dict(pretrained_weights, strict=False)

# Fine-tune on ABIDE
train_model(model, abide_data, fine_tune=True)
```

### 2. Ensemble Methods

Combine multiple models for better performance:

```python
# Train multiple models with different seeds
models = [train_model(seed=i) for i in range(5)]

# Ensemble prediction
predictions = [model.predict(X) for model in models]
final_prediction = np.mean(predictions, axis=0)
```

### 3. Test-Time Augmentation

Average predictions over augmented versions:

```python
# Apply multiple augmentations
augmented_samples = [augment(X) for _ in range(10)]

# Average predictions
predictions = [model(x) for x in augmented_samples]
final_prediction = torch.mean(torch.stack(predictions), dim=0)
```

## 🐛 Troubleshooting

### Out of Memory Error

```python
# Reduce batch size
config['batch_size'] = 16  # or 8

# Use gradient accumulation
accumulation_steps = 4
```

### Slow Training

```python
# Use GPU
device = torch.device('cuda')

# Reduce model size
config['hidden_dim'] = 128  # instead of 256

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler
```

### Poor Performance

```python
# Increase regularization
config['dropout'] = 0.5
config['weight_decay'] = 0.05

# Adjust learning rate
config['learning_rate'] = 5e-4

# More epochs
config['epochs'] = 300
```

## 📚 Citation

If you use this code, please cite:

```bibtex
@article{braingnn_multimodal,
  title={BrainGNN-Multimodal: Deep Learning for Autism Classification},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

MIT License - feel free to use for research and commercial purposes.

## 📧 Contact

For questions or issues, please open a GitHub issue or contact:
- Email: your.email@example.com
- GitHub: @yourusername

## 🙏 Acknowledgments

- ABIDE dataset: http://fcon_1000.projects.nitrc.org/indi/abide/
- PyTorch: https://pytorch.org/
- scikit-learn: https://scikit-learn.org/

---

**Happy Deep Learning! 🧠🤖**
