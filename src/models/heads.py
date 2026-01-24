import torch
from torch import nn


class PredictionHead(nn.Module):
    """
    Prediction head for valence/arousal.
    
    Args:
        input_dim: Input feature dimension (from LSTM)
        dropout: Dropout probability
        constrain_output: If True, constrain valence to [-2,2] and arousal to [0,2]
        verbose: Print shape information during forward pass
    
    Input: [B, S, input_dim]
    Output: [B, S, 2] (valence, arousal)
    """
    
    def __init__(self, input_dim, dropout=0.3, constrain_output=False, verbose=False):
        super().__init__()
        
        self.verbose = verbose
        self.constrain_output = constrain_output
        self.dropout = nn.Dropout(dropout)
        
        if constrain_output:
            self.valence_head = nn.Linear(input_dim, 1)
            self.arousal_head = nn.Linear(input_dim, 1)
            self.valence_activation = nn.Tanh()
            self.arousal_activation = nn.Sigmoid()
        else:
            self.fc = nn.Linear(input_dim, 2)
        
        if self.verbose:
            print(f"[PredictionHead] Initialized")
            print(f"  input_dim:        {input_dim}")
            print(f"  dropout:          {dropout}")
            print(f"  constrain_output: {constrain_output}")
            if constrain_output:
                print(f"  valence range:    [-2, 2] (tanh * 2)")
                print(f"  arousal range:    [0, 2] (sigmoid * 2)")
    
    def forward(self, x):
        """
        Args:
            x: [B, S, input_dim]
        
        Returns:
            predictions: [B, S, 2] (valence, arousal)
        """
        if self.verbose:
            print(f"\n  [PredictionHead] Forward pass")
            print(f"    Input x: {x.shape}")
        
        x = self.dropout(x)
        
        if self.constrain_output:
            v = self.valence_activation(self.valence_head(x)) * 2  # [B, S, 1] in [-2, 2]
            a = self.arousal_activation(self.arousal_head(x)) * 2  # [B, S, 1] in [0, 2]
            predictions = torch.cat([v, a], dim=-1)
            
            if self.verbose:
                print(f"    Valence: {v.shape}, range: [{v.min().item():.3f}, {v.max().item():.3f}]")
                print(f"    Arousal: {a.shape}, range: [{a.min().item():.3f}, {a.max().item():.3f}]")
        else:
            predictions = self.fc(x)
            
            if self.verbose:
                v_range = predictions[..., 0]
                a_range = predictions[..., 1]
                print(f"    Valence range: [{v_range.min().item():.3f}, {v_range.max().item():.3f}]")
                print(f"    Arousal range: [{a_range.min().item():.3f}, {a_range.max().item():.3f}]")
        
        if self.verbose:
            print(f"    Output: {predictions.shape}")
        
        return predictions