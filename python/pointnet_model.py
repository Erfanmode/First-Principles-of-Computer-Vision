# python/pointnet_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Core Network: PointNet-like Noise Predictor (ε_θ) ---
class PointCloudNoisePredictor(nn.Module):
    def __init__(self, n_points, point_dim=3, embed_dim=128, t_steps=1000):
        super().__init__()
        
        self.n_points = n_points
        self.point_dim = point_dim
        self.t_steps = t_steps
        layer_dim = 128
        
        # Time Step Embedding (for conditioning)
        self.time_embed = nn.Sequential(
            nn.Linear(1, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # 1. Local Feature Extraction (Per-point MLP)
        # TODO: Define the MLP layers for local feature extraction
        # Simplified PointNet-style shared MLP to lift points to high-dim features
        # Kernel size in Conv1d is 1, so basically the permutation of points does not matter
        self.local_mlp = nn.Sequential(
            nn.Conv1d(point_dim, layer_dim, 1),
            nn.BatchNorm1d(layer_dim),
            nn.GELU(),
            nn.Conv1d(layer_dim, layer_dim*2, 1),
            nn.Dropout(0.2),
            nn.BatchNorm1d(layer_dim*2),
            nn.GELU(),
            nn.Conv1d(layer_dim*2, layer_dim*4, 1),
            nn.BatchNorm1d(layer_dim*4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(layer_dim*4, embed_dim, 1),
        )

        # 2. Global Feature Extraction (Max Pooling)
        # TODO: Define the global feature extraction logic
        # Symmetric max pooling will be applied in forward() after loc2global_transform
        self.loc2global_transform = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim*2, 1),
            nn.BatchNorm1d(embed_dim*2),
            nn.GELU(),
            nn.Conv1d(embed_dim*2, embed_dim, 1)
        )

        # 3. Noise Prediction MLP
        # TODO: Define the MLP layers for noise prediction
        # Takes concatenated [per-point features, global feature, time embedding]
        self.noise_pred_mlp = nn.Sequential(
            nn.Conv1d(embed_dim * 3, layer_dim*8, 1),   # embed_dim × 3 channels
            nn.BatchNorm1d(layer_dim*8),
            nn.GELU(),
            nn.Conv1d(layer_dim*8, layer_dim*4, 1),
            nn.BatchNorm1d(layer_dim*4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(layer_dim*4, point_dim, 1),  # output: predicted noise ε per point
        )

    def forward(self, x, t):
        """
        Forward pass of the PointCloudNoisePredictor.
        
        Args:
            x (torch.Tensor): Input point cloud (B, N_POINTS, POINT_DIM).
            t (torch.Tensor): Time steps (B,).
        
        Returns:
            torch.Tensor: Predicted noise (B, N_POINTS, POINT_DIM).
        """
        
        # x: (B, N_POINTS, POINT_DIM)
        B, N, D = x.shape
        
        # TODO: Implement the forward pass logic here
        # (B, N, D) → (B, D, N) for Conv1d 
        x = x.transpose(1, 2)

        # Time embedding
        t = t.float().unsqueeze(1)              # (B, 1)
        t_emb = self.time_embed(t)               # (B, embed_dim)

        # 1. Point Feature Extraction: per-point MLP
        per_point_feat = self.local_mlp(x)        # (B, embed_dim, N)
        
        # 2. Global Feature Aggregation: max pooling (symmetric function)
        pre_global_feat = self.loc2global_transform(per_point_feat)      # (B, embed_dim, N)
        global_feat = torch.max(pre_global_feat, dim=2, keepdim=True)[0]  # (B, embed_dim, 1)
        global_feat = global_feat.expand(-1, -1, N)                      # (B, embed_dim, N)

        # Broadcast time embedding to all points
        t_emb = t_emb.unsqueeze(2).expand(-1, -1, N)                     # (B, embed_dim, N)

        # 3. Feature Combination and Noise Prediction
        combined = torch.cat([per_point_feat, global_feat, t_emb], dim=1)  # (B, 3×embed_dim, N)
        noise_pred = self.noise_pred_mlp(combined)                         # (B, point_dim, N)

        return noise_pred.transpose(1, 2)  # (B, N, point_dim)