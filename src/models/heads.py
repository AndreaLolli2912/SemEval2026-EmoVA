import torch
from torch import nn

class PredictionHead(nn.Module):
    """
    Prediction head for valence/arousal.
    
    Input: [B, S, input_dim]
    Output: [B, S, 2]
    """
    
    def __init__(self, input_dim, dropout=0.3, constrain_arousal=False):
        super().__init__()
        
        self.constrain_arousal = constrain_arousal
        self.dropout = nn.Dropout(dropout)
        
        if constrain_arousal:
            self.valence_head = nn.Linear(input_dim, 1)
            self.arousal_head = nn.Linear(input_dim, 1)
        else:
            self.fc = nn.Linear(input_dim, 2)
    
    def forward(self, x):
        """
        Args:
            x: [B, S, input_dim]
        
        Returns:
            predictions: [B, S, 2] (valence, arousal)
        """
        x = self.dropout(x)
        
        if self.constrain_arousal:
            v = self.valence_head(x)                     # [B, S, 1]
            a = torch.sigmoid(self.arousal_head(x)) * 2  # [B, S, 1] in [0, 2]
            return torch.cat([v, a], dim=-1)
        else:
            return self.fc(x)