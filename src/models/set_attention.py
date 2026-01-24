"""
Building Blocks for Efficient Attention Pooling

Based on "Set Transformer" (Lee et al., 2019)
Paper: https://arxiv.org/abs/1810.00825
"""

import torch
from torch import nn


class MAB(nn.Module):
    """
    Multihead Attention Block.
    
    Basic building block for ISAB and PMA.
    """
    
    def __init__(self, hidden_size, n_heads, dropout=0.1, verbose=False):
        super().__init__()
        
        self.verbose = verbose
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        
        self.attn = nn.MultiheadAttention(
            hidden_size, n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_size)
    
    def forward(self, Q, KV, mask=None):
        """
        Args:
            Q: [B, N, H] - queries
            KV: [B, M, H] - keys and values
            mask: [B, M] - True = ignore
        
        Returns:
            [B, N, H]
        """
        if self.verbose:
            print(f"    [MAB] Q: {Q.shape}, KV: {KV.shape}", end="")
            if mask is not None:
                print(f", mask: {mask.shape} (ignoring {mask.sum().item()} positions)\n")
            else:
                print(" (no mask)\n")
        
        attn_out, _ = self.attn(Q, KV, KV, key_padding_mask=mask)
        H = self.norm1(Q + attn_out)
        out = self.norm2(H + self.ffn(H))
        
        if self.verbose:
            print(f"    [MAB] Output: {out.shape}\n")
        
        return out


class ISAB(nn.Module):
    """
    Induced Set Attention Block.
    
    Efficient self-attention using inducing points as bottleneck.
    Complexity: O(T·m) instead of O(T²)
    """
    
    def __init__(self, hidden_size, n_heads, n_inducing=32, dropout=0.1, verbose=False):
        super().__init__()
        
        self.verbose = verbose
        self.n_inducing = n_inducing
        
        self.inducing = nn.Parameter(torch.randn(n_inducing, hidden_size))
        self.mab1 = MAB(hidden_size, n_heads, dropout, verbose=verbose)
        self.mab2 = MAB(hidden_size, n_heads, dropout, verbose=verbose)
        
        if self.verbose:
            print(f"[ISAB] Initialized with {n_inducing} inducing points\n")
    
    def forward(self, X, mask=None):
        """
        Args:
            X: [B, T, H] - input tokens
            mask: [B, T] - True = ignore
        
        Returns:
            [B, T, H] - enriched tokens
        """
        B = X.size(0)
        
        if self.verbose:
            print(f"\n  [ISAB] Forward pass")
            print(f"    Input X: {X.shape}\n")
        
        # Expand inducing points for batch
        I = self.inducing.unsqueeze(0).expand(B, -1, -1)
        
        if self.verbose:
            print(f"    Inducing points expanded: {I.shape}")
            print(f"    Step 1: Inducing points gather from tokens\n")
        
        # Step 1: Inducing points gather from tokens
        H = self.mab1(Q=I, KV=X, mask=mask)
        
        if self.verbose:
            print(f"    Compressed H: {H.shape}")
            print(f"    Step 2: Tokens retrieve from compressed\n")
        
        # Step 2: Tokens retrieve from compressed (no mask - H has no padding)
        out = self.mab2(Q=X, KV=H)
        
        if self.verbose:
            print(f"    Output: {out.shape}\n")
        
        return out


class PMA(nn.Module):
    """
    Pooling by Multihead Attention.
    
    Compresses variable-length input to fixed-size output.
    """
    
    def __init__(self, hidden_size, n_heads, n_seeds=1, dropout=0.1, verbose=False):
        super().__init__()
        
        self.verbose = verbose
        self.n_seeds = n_seeds
        
        self.seeds = nn.Parameter(torch.randn(n_seeds, hidden_size))
        self.mab = MAB(hidden_size, n_heads, dropout, verbose=verbose)
        
        if self.verbose:
            print(f"[PMA] Initialized with {n_seeds} seed vectors\n")
    
    def forward(self, X, mask=None):
        """
        Args:
            X: [B, T, H] - input tokens
            mask: [B, T] - True = ignore
        
        Returns:
            [B, K, H] - K summary vectors
        """
        B = X.size(0)
        
        if self.verbose:
            print(f"\n  [PMA] Forward pass")
            print(f"    Input X: {X.shape}\no")
        
        # Expand seeds for batch
        S = self.seeds.unsqueeze(0).expand(B, -1, -1)
        
        if self.verbose:
            print(f"    Seeds expanded: {S.shape}")
            print(f"    Seeds attend to tokens:\n")
        
        out = self.mab(Q=S, KV=X, mask=mask)
        
        if self.verbose:
            print(f"    Output (pooled): {out.shape}\n")
        
        return out