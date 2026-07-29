"""
Residual Domain Adapter — DynaCampus-FR Project
Novelty 3: Adapts low-resolution CCTV surveillance face embeddings to match high-resolution gallery prototypes.

z_adapted = z + f_theta(z)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDomainAdapter(nn.Module):
    """
    Lightweight MLP adapter that learns a residual transformation mapping
    surveillance domain embeddings closer to high-quality enrollment prototypes.
    """

    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual skip-connection and L2 normalization.
        x: shape (B, 512)
        returns: L2-normalized adapted embeddings (B, 512)
        """
        residual = x
        out = self.fc1(x)
        if out.shape[0] > 1:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        # Residual skip connection: z_adapted = normalize(z + f(z))
        adapted = residual + out
        return F.normalize(adapted, p=2, dim=1)
