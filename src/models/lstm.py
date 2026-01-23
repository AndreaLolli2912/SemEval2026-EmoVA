import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMEncoder(nn.Module):
    """
    LSTM encoder for sequential embeddings.
    
    Input: [B, S, input_dim]
    Output: [B, S, hidden_dim * num_directions]
    """
    
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, bidirectional=True, dropout=0.3, cls_dim=None):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # CLS projection: identity if dims match or not provided
        if cls_dim is not None and cls_dim != hidden_dim:
            self.cls_proj = nn.Linear(cls_dim, hidden_dim)
        else:
            self.cls_proj = nn.Identity()
        
        self.output_dim = hidden_dim * self.num_directions
    
    def forward(self, x, seq_lengths, init_hidden=None):
        """
        Args:
            x: [B, S, input_dim]
            seq_lengths: [B] actual lengths
            init_hidden: [B, cls_dim] optional CLS embedding to initialize hidden state
        
        Returns:
            output: [B, S, hidden_dim * num_directions]
        """
        hidden = None
        
        if init_hidden is not None:
            B = init_hidden.size(0)
            h_proj = self.cls_proj(init_hidden)  # [B, hidden_dim]
            
            # Expand for all layers and directions
            h_0 = h_proj.unsqueeze(0).expand(self.num_layers * self.num_directions, -1, -1)
            c_0 = torch.zeros_like(h_0)
            hidden = (h_0.contiguous(), c_0.contiguous())
        
        packed = pack_padded_sequence(
            x, seq_lengths.cpu(),
            batch_first=True, enforce_sorted=False
        )
        
        packed_out, _ = self.lstm(packed, hidden)
        output, _ = pad_packed_sequence(packed_out, batch_first=True)
        
        return output