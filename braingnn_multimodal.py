"""
BrainGNN-Multimodal: Advanced Deep Learning for Autism Classification
Using Multimodal Neuroimaging Data (fMRI + sMRI)

This implementation includes:
- Graph Neural Networks for fMRI connectivity
- Deep Neural Networks for sMRI features
- Multimodal fusion with cross-modal attention
- Domain adaptation for multi-site data
- Multi-task learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from typing import Optional, Tuple


# ============================================================================
# Graph Neural Network Components
# ============================================================================

class GraphConvolution(nn.Module):
    """
    Simple Graph Convolutional Layer
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Node features (batch_size, num_nodes, in_features)
            adj: Adjacency matrix (batch_size, num_nodes, num_nodes)
        Returns:
            output: Transformed features (batch_size, num_nodes, out_features)
        """
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT)
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3, 
                 alpha: float = 0.2, concat: bool = True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Node features (batch_size, num_nodes, in_features)
            adj: Adjacency matrix (batch_size, num_nodes, num_nodes)
        """
        batch_size, num_nodes, _ = h.size()
        
        # Linear transformation
        Wh = torch.matmul(h, self.W)  # (batch_size, num_nodes, out_features)
        
        # Attention mechanism
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        
        # Mask attention weights using adjacency matrix
        # Use a more stable mask value
        zero_vec = -1e9 * torch.ones_like(e)
        attention = torch.where(adj != 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Apply attention to features
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, out_features = Wh.size()
        
        # Repeat features for all pairs
        Wh_repeated_in_chunks = Wh.repeat_interleave(num_nodes, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, num_nodes, 1)
        
        # Concatenate
        all_combinations_matrix = torch.cat(
            [Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2
        )
        
        return all_combinations_matrix.view(batch_size, num_nodes, num_nodes, 2 * out_features)


class GraphPooling(nn.Module):
    """
    Top-K Graph Pooling Layer
    """
    def __init__(self, in_features: int, ratio: float = 0.5):
        super(GraphPooling, self).__init__()
        self.in_features = in_features
        self.ratio = ratio
        self.score_layer = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node features (batch_size, num_nodes, in_features)
            adj: Adjacency matrix (batch_size, num_nodes, num_nodes)
        Returns:
            x_pooled: Pooled features
            adj_pooled: Pooled adjacency matrix
        """
        batch_size, num_nodes, _ = x.size()
        
        # Compute node scores
        scores = self.score_layer(x).squeeze(-1)  # (batch_size, num_nodes)
        
        # Select top-k nodes
        k = max(int(num_nodes * self.ratio), 1)
        _, idx = torch.topk(scores, k, dim=1)
        
        # Pool features
        x_pooled = torch.gather(
            x, 1, idx.unsqueeze(-1).expand(-1, -1, self.in_features)
        )
        
        # Pool adjacency matrix
        adj_pooled = torch.gather(
            adj, 1, idx.unsqueeze(-1).expand(-1, -1, num_nodes)
        )
        adj_pooled = torch.gather(
            adj_pooled, 2, idx.unsqueeze(1).expand(-1, k, -1)
        )
        
        return x_pooled, adj_pooled


# ============================================================================
# fMRI Graph Neural Network Branch
# ============================================================================

class fMRIGraphBranch(nn.Module):
    """
    Graph Neural Network branch for fMRI connectivity matrices
    """
    def __init__(self, num_nodes: int = 200, hidden_dim: int = 256, 
                 num_layers: int = 3, dropout: float = 0.3):
        super(fMRIGraphBranch, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # Graph construction parameters
        self.edge_threshold = 0.2  # Lowered to allow more information flow
        
        # Graph convolutional layers
        self.gcn1 = GraphConvolution(num_nodes, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn3 = GraphConvolution(hidden_dim, hidden_dim)
        
        # Graph attention layer
        self.gat = GraphAttentionLayer(hidden_dim, hidden_dim, dropout=dropout, concat=False)
        
        # Graph pooling
        self.pool = GraphPooling(hidden_dim, ratio=0.5)
        
        # Batch normalization (should be applied to hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # Self-attention for global features
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout)
        
        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)

    def construct_graph(self, connectivity_matrix: torch.Tensor) -> torch.Tensor:
        """
        Convert connectivity matrix to adjacency matrix
        Args:
            connectivity_matrix: (batch_size, num_nodes, num_nodes)
        Returns:
            adj: Adjacency matrix with self-loops and normalization
        """
        # Threshold weak connections
        adj = connectivity_matrix.clone()
        adj = torch.where(torch.abs(adj) > self.edge_threshold, adj, torch.zeros_like(adj))
        
        # Add self-loops
        batch_size = adj.size(0)
        eye = torch.eye(self.num_nodes, device=adj.device).unsqueeze(0).repeat(batch_size, 1, 1)
        adj = adj + eye
        
        # Normalize adjacency matrix (symmetric normalization)
        # Add epsilon for numerical stability
        deg = torch.sum(torch.abs(adj), dim=2) + 1e-8
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.
        
        # D^(-1/2) * A * D^(-1/2)
        adj_normalized = deg_inv_sqrt.unsqueeze(2) * adj * deg_inv_sqrt.unsqueeze(1)
        
        # Clamp values to prevent explosion
        adj_normalized = torch.clamp(adj_normalized, min=-10, max=10)
        
        return adj_normalized

    def forward(self, connectivity_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            connectivity_matrix: (batch_size, num_nodes, num_nodes)
        Returns:
            output: (batch_size, 128)
            attention_weights: For visualization
        """
        batch_size = connectivity_matrix.size(0)
        
        # Construct graph
        adj = self.construct_graph(connectivity_matrix)
        
        # Use connectivity matrix as initial node features
        x = connectivity_matrix  # (batch_size, num_nodes, num_nodes)
        
        # GCN layers
        x = self.gcn1(x, adj)
        # BatchNorm1d expects (batch, channels, length)
        # Our x is (batch, num_nodes, hidden_dim)
        # So we transpose to (batch, hidden_dim, num_nodes)
        x = x.transpose(1, 2)
        x = self.bn1(x)
        x = x.transpose(1, 2)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.gcn2(x, adj)
        x = x.transpose(1, 2)
        x = self.bn2(x)
        x = x.transpose(1, 2)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.gcn3(x, adj)
        x = x.transpose(1, 2)
        x = self.bn3(x)
        x = x.transpose(1, 2)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Graph attention
        x = self.gat(x, adj)
        
        # Graph pooling
        x_pooled, adj_pooled = self.pool(x, adj)
        
        # Global pooling (mean over nodes)
        x_global = torch.mean(x_pooled, dim=1)  # (batch_size, hidden_dim)
        
        # Self-attention for capturing global dependencies
        x_seq = x_pooled.transpose(0, 1)  # (num_nodes, batch_size, hidden_dim)
        attn_output, attn_weights = self.self_attention(x_seq, x_seq, x_seq)
        attn_output = attn_output.transpose(0, 1)  # (batch_size, num_nodes, hidden_dim)
        
        # Combine global and attention features
        x_combined = x_global + torch.mean(attn_output, dim=1)
        
        # Output projection
        output = self.fc(x_combined)
        
        return output, attn_weights


# ============================================================================
# sMRI Deep Neural Network Branch
# ============================================================================

class ResidualBlock(nn.Module):
    """
    Residual block with batch normalization
    """
    def __init__(self, dim: int, dropout: float = 0.3):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class sMRIBranch(nn.Module):
    """
    Deep Neural Network branch for sMRI features
    """
    def __init__(self, input_dim: int = 2500, hidden_dim: int = 512, 
                 num_heads: int = 8, dropout: float = 0.3):
        super(sMRIBranch, self).__init__()
        
        # Feature embedding
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # Residual blocks
        self.res_block1 = ResidualBlock(hidden_dim, dropout)
        self.res_block2 = ResidualBlock(hidden_dim, dropout)
        
        # Feature attention (channel-wise)
        self.feature_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: sMRI features (batch_size, input_dim)
        Returns:
            output: (batch_size, 128)
            attention_weights: For visualization
        """
        # Feature embedding
        x = self.embedding(x)  # (batch_size, hidden_dim)
        
        # Self-attention (treat features as sequence)
        x_seq = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        x_seq = x_seq.transpose(0, 1)  # (1, batch_size, hidden_dim)
        attn_output, attn_weights = self.attention(x_seq, x_seq, x_seq)
        attn_output = attn_output.transpose(0, 1).squeeze(1)  # (batch_size, hidden_dim)
        
        # Residual connection
        x = self.attention_norm(x + attn_output)
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        # Feature attention (channel-wise attention)
        attention_weights_channel = self.feature_attention(x)
        x = x * attention_weights_channel
        
        # Output projection
        output = self.fc(x)
        
        return output, attention_weights_channel


# ============================================================================
# Phenotypic Embedding Branch
# ============================================================================

class PhenotypicBranch(nn.Module):
    """
    Embedding branch for phenotypic data (age, gender, site)
    """
    def __init__(self, num_sites: int = 20, age_dim: int = 16, 
                 gender_dim: int = 8):
        super(PhenotypicBranch, self).__init__()
        
        # Site embedding (for domain adaptation)
        self.site_embedding = nn.Embedding(num_sites, 32)
        
        # Age encoding (continuous variable)
        self.age_encoder = nn.Sequential(
            nn.Linear(1, age_dim),
            nn.ReLU(),
            nn.Linear(age_dim, age_dim)
        )
        
        # Gender embedding (categorical)
        self.gender_embedding = nn.Embedding(3, gender_dim)  # 0=unknown, 1=M, 2=F
        
        # Combine all phenotypic features
        total_dim = 32 + age_dim + gender_dim
        self.fc = nn.Sequential(
            nn.Linear(total_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64)
        )

    def forward(self, site: torch.Tensor, age: torch.Tensor, 
                gender: torch.Tensor) -> torch.Tensor:
        """
        Args:
            site: (batch_size,) - site indices
            age: (batch_size, 1) - age values
            gender: (batch_size,) - gender indices
        Returns:
            output: (batch_size, 64)
        """
        # Embed each feature
        site_emb = self.site_embedding(site)
        age_emb = self.age_encoder(age)
        gender_emb = self.gender_embedding(gender)
        
        # Concatenate all features
        combined = torch.cat([site_emb, age_emb, gender_emb], dim=1)
        
        # Project to output dimension
        output = self.fc(combined)
        
        return output


# ============================================================================
# Multimodal Fusion Layer
# ============================================================================

class MultimodalFusion(nn.Module):
    """
    Multimodal fusion with cross-modal attention and bilinear pooling
    """
    def __init__(self, fmri_dim: int = 128, smri_dim: int = 128, 
                 pheno_dim: int = 64, dropout: float = 0.4):
        super(MultimodalFusion, self).__init__()
        
        # Cross-modal attention (fMRI attends to sMRI)
        self.cross_attention_f2s = nn.MultiheadAttention(fmri_dim, num_heads=4, dropout=dropout)
        self.cross_attention_s2f = nn.MultiheadAttention(smri_dim, num_heads=4, dropout=dropout)
        
        # Bilinear pooling for interaction modeling
        self.bilinear = nn.Bilinear(fmri_dim, smri_dim, 128)
        
        # Fusion layers
        total_dim = fmri_dim + smri_dim + 128 + pheno_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, fmri_features: torch.Tensor, smri_features: torch.Tensor, 
                pheno_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fmri_features: (batch_size, fmri_dim)
            smri_features: (batch_size, smri_dim)
            pheno_features: (batch_size, pheno_dim)
        Returns:
            fused_features: (batch_size, 128)
        """
        # Prepare for cross-attention (add sequence dimension)
        fmri_seq = fmri_features.unsqueeze(0)  # (1, batch_size, fmri_dim)
        smri_seq = smri_features.unsqueeze(0)  # (1, batch_size, smri_dim)
        
        # Cross-modal attention
        fmri_attended, _ = self.cross_attention_f2s(fmri_seq, smri_seq, smri_seq)
        smri_attended, _ = self.cross_attention_s2f(smri_seq, fmri_seq, fmri_seq)
        
        fmri_attended = fmri_attended.squeeze(0)
        smri_attended = smri_attended.squeeze(0)
        
        # Bilinear pooling (second-order interactions)
        bilinear_features = self.bilinear(fmri_features, smri_features)
        
        # Concatenate all features
        combined = torch.cat([
            fmri_attended,
            smri_attended,
            bilinear_features,
            pheno_features
        ], dim=1)
        
        # Fusion
        fused_features = self.fusion(combined)
        
        return fused_features


# ============================================================================
# Classification Head with Auxiliary Tasks
# ============================================================================

class ClassificationHead(nn.Module):
    """
    Classification head with auxiliary tasks for multi-task learning
    """
    def __init__(self, input_dim: int = 128, num_sites: int = 20, dropout: float = 0.5):
        super(ClassificationHead, self).__init__()
        
        # Main classification task (ASD vs TD)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        
        # Auxiliary task 1: Site prediction (for domain adaptation)
        self.site_classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_sites)
        )
        
        # Auxiliary task 2: Age regression (deconfounding)
        self.age_regressor = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Fused features (batch_size, input_dim)
        Returns:
            class_logits: (batch_size, 2) - ASD vs TD
            site_logits: (batch_size, num_sites) - Site prediction
            age_pred: (batch_size, 1) - Age prediction
        """
        class_logits = self.classifier(x)
        site_logits = self.site_classifier(x)
        age_pred = self.age_regressor(x)
        
        return class_logits, site_logits, age_pred


# ============================================================================
# Complete BrainGNN-Multimodal Model
# ============================================================================

class BrainGNNMultimodal(nn.Module):
    """
    Complete multimodal deep learning model for autism classification
    """
    def __init__(self, 
                 num_nodes: int = 200,
                 smri_dim: int = 2500,
                 num_sites: int = 20,
                 hidden_dim: int = 256,
                 dropout: float = 0.3):
        super(BrainGNNMultimodal, self).__init__()
        
        # fMRI branch
        self.fmri_branch = fMRIGraphBranch(
            num_nodes=num_nodes,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # sMRI branch
        self.smri_branch = sMRIBranch(
            input_dim=smri_dim,
            hidden_dim=512,
            dropout=dropout
        )
        
        # Phenotypic branch
        self.pheno_branch = PhenotypicBranch(num_sites=num_sites)
        
        # Multimodal fusion
        self.fusion = MultimodalFusion(
            fmri_dim=128,
            smri_dim=128,
            pheno_dim=64,
            dropout=0.4
        )
        
        # Classification head
        self.classifier = ClassificationHead(
            input_dim=128,
            num_sites=num_sites,
            dropout=0.5
        )

    def forward(self, fmri_data: torch.Tensor, smri_data: torch.Tensor,
                site: torch.Tensor, age: torch.Tensor, gender: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass through the entire model
        
        Args:
            fmri_data: (batch_size, num_nodes, num_nodes) - Connectivity matrices
            smri_data: (batch_size, smri_dim) - sMRI features
            site: (batch_size,) - Site indices
            age: (batch_size, 1) - Age values
            gender: (batch_size,) - Gender indices
            
        Returns:
            class_logits: (batch_size, 2) - Classification logits
            site_logits: (batch_size, num_sites) - Site prediction logits
            age_pred: (batch_size, 1) - Age predictions
            attention_dict: Dictionary of attention weights for visualization
        """
        # Process each modality
        fmri_features, fmri_attention = self.fmri_branch(fmri_data)
        smri_features, smri_attention = self.smri_branch(smri_data)
        pheno_features = self.pheno_branch(site, age, gender)
        
        # Multimodal fusion
        fused_features = self.fusion(fmri_features, smri_features, pheno_features)
        
        # Classification with auxiliary tasks
        class_logits, site_logits, age_pred = self.classifier(fused_features)
        
        # Collect attention weights for visualization
        attention_dict = {
            'fmri_attention': fmri_attention,
            'smri_attention': smri_attention
        }
        
        return class_logits, site_logits, age_pred, attention_dict

    def get_embeddings(self, fmri_data: torch.Tensor, smri_data: torch.Tensor,
                      site: torch.Tensor, age: torch.Tensor, gender: torch.Tensor) -> torch.Tensor:
        """
        Get fused embeddings for visualization or further analysis
        """
        fmri_features, _ = self.fmri_branch(fmri_data)
        smri_features, _ = self.smri_branch(smri_data)
        pheno_features = self.pheno_branch(site, age, gender)
        fused_features = self.fusion(fmri_features, smri_features, pheno_features)
        return fused_features


# ============================================================================
# Model Factory
# ============================================================================

def create_model(config: dict) -> BrainGNNMultimodal:
    """
    Factory function to create model with configuration
    
    Args:
        config: Dictionary with model configuration
        
    Returns:
        model: BrainGNNMultimodal instance
    """
    model = BrainGNNMultimodal(
        num_nodes=config.get('num_nodes', 200),
        smri_dim=config.get('smri_dim', 2500),
        num_sites=config.get('num_sites', 20),
        hidden_dim=config.get('hidden_dim', 256),
        dropout=config.get('dropout', 0.3)
    )
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing BrainGNN-Multimodal Model...")
    
    # Create dummy data
    batch_size = 4
    num_nodes = 200
    smri_dim = 2500
    num_sites = 20
    
    fmri_data = torch.randn(batch_size, num_nodes, num_nodes)
    smri_data = torch.randn(batch_size, smri_dim)
    site = torch.randint(0, num_sites, (batch_size,))
    age = torch.randn(batch_size, 1) * 10 + 20  # Age around 20
    gender = torch.randint(0, 2, (batch_size,))
    # Create model
    config = {
        'num_nodes': num_nodes,
        'smri_dim': smri_dim,
        'num_sites': num_sites,
        'hidden_dim': 256,
        'dropout': 0.3
    }
    model = create_model(config)
    
    # Forward pass
    class_logits, site_logits, age_pred, attention_dict = model(
        fmri_data, smri_data, site, age, gender
    )
    
    print(f"Class logits shape: {class_logits.shape}")
    print(f"Site logits shape: {site_logits.shape}")
    print(f"Age prediction shape: {age_pred.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nModel test passed!")
