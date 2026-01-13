"""
BrainGNN-Multimodal: Advanced Deep Learning for Autism Classification
Using Multimodal Neuroimaging Data (fMRI + sMRI)

IMPROVED VERSION:
- Adaptive Graph Construction (Top-K Sparsification)
- Residual GNN Connections
- Gated Multimodal Fusion
- Enhanced sMRI Branch with Residual Blocks
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
    Graph Convolutional Layer with Residual Connection support
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
        
        # Residual projection if dimensions differ
        if in_features != out_features:
            self.res_proj = nn.Linear(in_features, out_features)
        else:
            self.res_proj = None
            
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
        """
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        
        if self.bias is not None:
            output = output + self.bias
            
        # Add residual connection
        if self.res_proj:
            res = self.res_proj(input)
        else:
            res = input
            
        return F.relu(output + res)


class GraphAttentionLayer(nn.Module):
    """
    Improved Graph Attention Layer (GATv2 style)
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
        batch_size, num_nodes, _ = h.size()
        Wh = torch.matmul(h, self.W)
        
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        
        zero_vec = -1e9 * torch.ones_like(e)
        attention = torch.where(adj != 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, out_features = Wh.size()
        Wh_repeated_in_chunks = Wh.repeat_interleave(num_nodes, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, num_nodes, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
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
        batch_size, num_nodes, _ = x.size()
        scores = self.score_layer(x).squeeze(-1)
        k = max(int(num_nodes * self.ratio), 1)
        _, idx = torch.topk(scores, k, dim=1)
        
        x_pooled = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, self.in_features))
        adj_pooled = torch.gather(adj, 1, idx.unsqueeze(-1).expand(-1, -1, num_nodes))
        adj_pooled = torch.gather(adj_pooled, 2, idx.unsqueeze(1).expand(-1, k, -1))
        
        return x_pooled, adj_pooled


# ============================================================================
# fMRI Graph Neural Network Branch
# ============================================================================

class fMRIGraphBranch(nn.Module):
    """
    Improved GNN branch with Adaptive Graph Construction
    """
    def __init__(self, num_nodes: int = 200, hidden_dim: int = 256, 
                 num_layers: int = 3, dropout: float = 0.3):
        super(fMRIGraphBranch, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # Adaptive sparsification: keep top 20% of edges
        self.keep_ratio = 0.15 
        
        # Graph convolutional layers with residual support
        self.gcn1 = GraphConvolution(num_nodes, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn3 = GraphConvolution(hidden_dim, hidden_dim)
        
        self.gat = GraphAttentionLayer(hidden_dim, hidden_dim, dropout=dropout, concat=False)
        self.pool = GraphPooling(hidden_dim, ratio=0.5)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def construct_graph(self, connectivity_matrix: torch.Tensor) -> torch.Tensor:
        """
        Adaptive Graph Construction using Top-K Sparsification
        """
        batch_size = connectivity_matrix.size(0)
        adj = connectivity_matrix.clone()
        
        # Absolute values for thresholding
        abs_adj = torch.abs(adj)
        
        # Find threshold for top-k edges per subject
        k = int(self.num_nodes * self.num_nodes * self.keep_ratio)
        if k > 0:
            flat_adj = abs_adj.view(batch_size, -1)
            thresholds, _ = torch.topk(flat_adj, k, dim=1)
            min_thresholds = thresholds[:, -1].view(batch_size, 1, 1)
            adj = torch.where(abs_adj >= min_thresholds, adj, torch.zeros_like(adj))
        
        # Add self-loops
        eye = torch.eye(self.num_nodes, device=adj.device).unsqueeze(0).repeat(batch_size, 1, 1)
        adj = adj + eye
        
        # Symmetric normalization
        deg = torch.sum(torch.abs(adj), dim=2) + 1e-8
        deg_inv_sqrt = torch.pow(deg, -0.5)
        adj_normalized = deg_inv_sqrt.unsqueeze(2) * adj * deg_inv_sqrt.unsqueeze(1)
        
        return adj_normalized

    def forward(self, connectivity_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = connectivity_matrix.size(0)
        adj = self.construct_graph(connectivity_matrix)
        x = connectivity_matrix 
        
        # GCN layers with residual connections (handled inside GraphConvolution)
        x = self.gcn1(x, adj)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)
        
        x = self.gcn2(x, adj)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)
        
        x = self.gcn3(x, adj)
        x = self.bn3(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)
        
        x = self.gat(x, adj)
        x_pooled, _ = self.pool(x, adj)
        
        # Global features
        x_global = torch.mean(x_pooled, dim=1)
        
        # Self-attention
        x_seq = x_pooled.transpose(0, 1)
        attn_output, attn_weights = self.self_attention(x_seq, x_seq, x_seq)
        attn_output = attn_output.transpose(0, 1)
        
        x_combined = x_global + torch.mean(attn_output, dim=1)
        output = self.fc(x_combined)
        
        return output, attn_weights


# ============================================================================
# sMRI Deep Neural Network Branch
# ============================================================================

class ResidualBlock(nn.Module):
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
        return F.relu(out)


class sMRIBranch(nn.Module):
    def __init__(self, input_dim: int = 2500, hidden_dim: int = 512, 
                 num_heads: int = 8, dropout: float = 0.3):
        super(sMRIBranch, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.res_block1 = ResidualBlock(hidden_dim, dropout)
        self.res_block2 = ResidualBlock(hidden_dim, dropout)
        
        self.feature_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Sigmoid()
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(x)
        x_seq = x.unsqueeze(0)
        attn_output, _ = self.attention(x_seq, x_seq, x_seq)
        attn_output = attn_output.squeeze(0)
        x = self.attention_norm(x + attn_output)
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        weights = self.feature_attention(x)
        x = x * weights
        output = self.fc(x)
        return output, weights


# ============================================================================
# Phenotypic Embedding Branch
# ============================================================================

class PhenotypicBranch(nn.Module):
    def __init__(self, num_sites: int = 20, age_dim: int = 16, gender_dim: int = 8):
        super(PhenotypicBranch, self).__init__()
        self.site_embedding = nn.Embedding(num_sites, 32)
        self.age_encoder = nn.Sequential(
            nn.Linear(1, age_dim),
            nn.ReLU(),
            nn.Linear(age_dim, age_dim)
        )
        self.gender_embedding = nn.Embedding(3, gender_dim)
        total_dim = 32 + age_dim + gender_dim
        self.fc = nn.Sequential(
            nn.Linear(total_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64)
        )

    def forward(self, site: torch.Tensor, age: torch.Tensor, gender: torch.Tensor) -> torch.Tensor:
        site_emb = self.site_embedding(site)
        age_emb = self.age_encoder(age)
        gender_emb = self.gender_embedding(gender)
        combined = torch.cat([site_emb, age_emb, gender_emb], dim=1)
        return self.fc(combined)


# ============================================================================
# Improved Multimodal Fusion Layer
# ============================================================================

class MultimodalFusion(nn.Module):
    """
    Gated Multimodal Fusion with Cross-Attention
    """
    def __init__(self, fmri_dim: int = 128, smri_dim: int = 128, 
                 pheno_dim: int = 64, dropout: float = 0.4):
        super(MultimodalFusion, self).__init__()
        self.cross_attention_f2s = nn.MultiheadAttention(fmri_dim, num_heads=4, dropout=dropout)
        self.cross_attention_s2f = nn.MultiheadAttention(smri_dim, num_heads=4, dropout=dropout)
        self.bilinear = nn.Bilinear(fmri_dim, smri_dim, 128)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(fmri_dim + smri_dim + 128 + pheno_dim, 4),
            nn.Softmax(dim=1)
        )
        
        total_dim = fmri_dim + smri_dim + 128 + pheno_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, fmri_features: torch.Tensor, smri_features: torch.Tensor, 
                pheno_features: torch.Tensor) -> torch.Tensor:
        fmri_seq = fmri_features.unsqueeze(0)
        smri_seq = smri_features.unsqueeze(0)
        
        fmri_attended, _ = self.cross_attention_f2s(fmri_seq, smri_seq, smri_seq)
        smri_attended, _ = self.cross_attention_s2f(smri_seq, fmri_seq, fmri_seq)
        
        fmri_attended = fmri_attended.squeeze(0)
        smri_attended = smri_attended.squeeze(0)
        bilinear_features = self.bilinear(fmri_features, smri_features)
        
        # Concatenate and apply gating
        combined = torch.cat([fmri_attended, smri_attended, bilinear_features, pheno_features], dim=1)
        gates = self.gate(combined)
        
        # Apply gates to each component
        f_weighted = fmri_attended * gates[:, 0:1]
        s_weighted = smri_attended * gates[:, 1:2]
        b_weighted = bilinear_features * gates[:, 2:3]
        p_weighted = pheno_features * gates[:, 3:4]
        
        weighted_combined = torch.cat([f_weighted, s_weighted, b_weighted, p_weighted], dim=1)
        return self.fusion(weighted_combined)


# ============================================================================
# Classification Head
# ============================================================================

class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int = 128, num_sites: int = 20, dropout: float = 0.5):
        super(ClassificationHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        self.site_classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_sites)
        )
        self.age_regressor = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.classifier(x), self.site_classifier(x), self.age_regressor(x)


# ============================================================================
# Complete BrainGNN-Multimodal Model
# ============================================================================

class BrainGNNMultimodal(nn.Module):
    def __init__(self, num_nodes: int = 200, smri_dim: int = 2500,
                 num_sites: int = 20, hidden_dim: int = 256, dropout: float = 0.3):
        super(BrainGNNMultimodal, self).__init__()
        self.fmri_branch = fMRIGraphBranch(num_nodes, hidden_dim, dropout=dropout)
        self.smri_branch = sMRIBranch(smri_dim, hidden_dim=512, dropout=dropout)
        self.pheno_branch = PhenotypicBranch(num_sites)
        self.fusion = MultimodalFusion(128, 128, 64, dropout=0.4)
        self.classifier = ClassificationHead(256, num_sites, dropout=0.5)

    def forward(self, fmri_data, smri_data, site, age, gender):
        fmri_feat, fmri_attn = self.fmri_branch(fmri_data)
        smri_feat, smri_attn = self.smri_branch(smri_data)
        pheno_feat = self.pheno_branch(site, age, gender)
        fused_feat = self.fusion(fmri_feat, smri_feat, pheno_feat)
        class_logits, site_logits, age_pred = self.classifier(fused_feat)
        return class_logits, site_logits, age_pred, {'fmri_attention': fmri_attn, 'smri_attention': smri_attn}

def create_model(config: dict) -> BrainGNNMultimodal:
    return BrainGNNMultimodal(
        num_nodes=config.get('num_nodes', 200),
        smri_dim=config.get('smri_dim', 2500),
        num_sites=config.get('num_sites', 20),
        hidden_dim=config.get('hidden_dim', 256),
        dropout=config.get('dropout', 0.3)
    )
