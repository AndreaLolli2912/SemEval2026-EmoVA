import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMEncoder(nn.Module):
    """
    LSTM encoder for sequential embeddings.
    
    Processes document sequences with packing for efficiency.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        bidirectional: Whether to use bidirectional LSTM
        dropout: Dropout probability (applied between layers if num_layers > 1)
        verbose: Print shape information during forward pass
    
    Input: 
        x: [B, S, input_dim] - document embeddings
        seq_lengths: [B] - actual sequence lengths
    
    Output: 
        [B, S, hidden_dim * num_directions]
    """
    
    def __init__(
        self, 
        input_dim, 
        hidden_dim=256, 
        num_layers=2, 
        bidirectional=True, 
        dropout=0.3,
        verbose=False
    ):
        super().__init__()
        
        self.verbose = verbose
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.output_dim = hidden_dim * self.num_directions
        
        if self.verbose:
            print(f"[LSTMEncoder] Initialized")
            print(f"  input_dim:     {input_dim}")
            print(f"  hidden_dim:    {hidden_dim}")
            print(f"  num_layers:    {num_layers}")
            print(f"  bidirectional: {bidirectional}")
            print(f"  output_dim:    {self.output_dim}")
    
    def forward(self, x, seq_lengths):
        """
        Args:
            x: [B, S, input_dim] - document embeddings
            seq_lengths: [B] - actual sequence lengths per user
        
        Returns:
            output: [B, S, hidden_dim * num_directions]
        """
        B, S, D = x.shape
        
        if self.verbose:
            print(f"\n  [LSTMEncoder] Forward pass")
            print(f"    Input x: {x.shape} (B={B}, S={S}, D={D})")
            print(f"    seq_lengths: {seq_lengths.tolist()}")
            print(f"    total valid timesteps: {seq_lengths.sum().item()}")
        
        # Pack sequences (efficient - skips padding)
        packed = pack_padded_sequence(
            x, seq_lengths.cpu(),
            batch_first=True, 
            enforce_sorted=False
        )
        
        if self.verbose:
            print(f"    Packed data shape: {packed.data.shape}")
            print(f"    Packed batch_sizes: {packed.batch_sizes[:5].tolist()}..." if len(packed.batch_sizes) > 5 else f"    Packed batch_sizes: {packed.batch_sizes.tolist()}")
        
        # LSTM forward
        packed_out, (h_n, c_n) = self.lstm(packed)
        
        if self.verbose:
            print(f"    h_n shape: {h_n.shape} (num_layers*directions, B, hidden)")
            print(f"    c_n shape: {c_n.shape}")
        
        # Unpack back to padded
        output, output_lengths = pad_packed_sequence(packed_out, batch_first=True)
        
        if self.verbose:
            print(f"    Output (unpacked): {output.shape}")
            print(f"    Output lengths: {output_lengths.tolist()}")
        
        return output