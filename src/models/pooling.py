"""
Building Blocks for Efficient Attention Pooling

Based on "Set Transformer" (Lee et al., 2019)
Paper: https://arxiv.org/abs/1810.00825

Components:
    - MAB: Multihead Attention Block (basic building block)
    - ISAB: Induced Set Attention Block (efficient self-attention)
    - PMA: Pooling by Multihead Attention (compress to fixed size)

Typical usage:
    tokens [B, T, H] → ISAB → [B, T, H] → PMA → [B, K, H]
    
    Where:
        B = batch size
        T = number of tokens (e.g., 512)
        H = hidden dimension (e.g., 768)
        K = number of output vectors (e.g., 4)
"""

import torch
from torch import nn


class MAB(nn.Module):
    """
    Multihead Attention Block.
    
    Basic building block for ISAB and PMA. Performs:
        1. Cross-attention: Q attends to KV
        2. Residual connection + LayerNorm
        3. Feedforward network
        4. Residual connection + LayerNorm
    
    Args:
        hidden_size: Embedding dimension (H)
        n_heads: Number of attention heads
        dropout: Dropout probability
    
    Input:
        Q: [B, N, H] - queries (what's asking)
        KV: [B, M, H] - keys/values (what's being attended to)
        mask: [B, M] - optional, True for positions to ignore
    
    Output:
        [B, N, H] - same shape as Q
    
    Example:
        >>> mab = MAB(hidden_size=768, n_heads=8)
        >>> Q = torch.randn(2, 4, 768)    # 4 queries
        >>> KV = torch.randn(2, 512, 768) # 512 tokens
        >>> out = mab(Q, KV)              # [2, 4, 768]
    """
    
    def __init__(self, hidden_size, n_heads, dropout=0.1):
        super().__init__()
        
        # Attention layer
        self.attn = nn.MultiheadAttention(
            hidden_size, n_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        
        # Feedforward network (expand then contract)
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
            KV: [B, M, H] - keys and values (same source)
            mask: [B, M] - True = ignore this position
        
        Returns:
            [B, N, H] - output (same shape as Q)
        """
        # Attention: Q asks, KV answers
        # key_padding_mask: True means "ignore this token"
        attn_out, _ = self.attn(Q, KV, KV, key_padding_mask=mask)
        
        # Residual + Norm
        H = self.norm1(Q + attn_out)
        
        # FFN + Residual + Norm
        return self.norm2(H + self.ffn(H))


class ISAB(nn.Module):
    """
    Induced Set Attention Block.
    
    Efficient self-attention using inducing points as bottleneck.
    Instead of O(T²) complexity, achieves O(T·m) where m << T.
    
    How it works:
        1. Inducing points gather info from all tokens (compress)
        2. Tokens retrieve info from inducing points (broadcast)
    
    Result: Tokens become "aware" of each other through the bottleneck.
    
    Args:
        hidden_size: Embedding dimension (H)
        n_heads: Number of attention heads
        n_inducing: Number of inducing points (m) - the bottleneck size
        dropout: Dropout probability
    
    Input:
        X: [B, T, H] - token embeddings
        mask: [B, T] - optional, True for positions to ignore
    
    Output:
        [B, T, H] - same shape as input, but enriched with global context
    
    Complexity:
        Normal self-attention: O(T²) = O(512²) = 262,144
        ISAB with m=32: O(T·m) = O(512·32) = 16,384 (16x cheaper)
    
    Example:
        >>> isab = ISAB(hidden_size=768, n_heads=8, n_inducing=32)
        >>> tokens = torch.randn(2, 512, 768)
        >>> enriched = isab(tokens)  # [2, 512, 768]
    """
    
    def __init__(self, hidden_size, n_heads, n_inducing=32, dropout=0.1):
        super().__init__()
        
        # Learned inducing points (the bottleneck)
        # These learn to specialize: some focus on sentiment, some on topics, etc.
        self.inducing = nn.Parameter(torch.randn(n_inducing, hidden_size))
        
        # Two MAB blocks for the two steps
        self.mab1 = MAB(hidden_size, n_heads, dropout)  # compress
        self.mab2 = MAB(hidden_size, n_heads, dropout)  # broadcast
    
    def forward(self, X, mask=None):
        """
        Args:
            X: [B, T, H] - input tokens
            mask: [B, T] - True = ignore this token
        
        Returns:
            [B, T, H] - enriched tokens (same shape as input)
        """
        B = X.size(0)
        
        # Expand inducing points for batch: [m, H] → [B, m, H]
        I = self.inducing.unsqueeze(0).expand(B, -1, -1)
        
        # Step 1: Inducing points gather from tokens
        # [B, m, H] attends to [B, T, H] → [B, m, H]
        H = self.mab1(Q=I, KV=X, mask=mask)
        
        # Step 2: Tokens retrieve from compressed representation
        # [B, T, H] attends to [B, m, H] → [B, T, H]
        # Note: no mask here because H has no padding
        return self.mab2(Q=X, KV=H)


class PMA(nn.Module):
    """
    Pooling by Multihead Attention.
    
    Compresses variable-length input to fixed-size output using learned seeds.
    Each seed learns to focus on different aspects of the input.
    
    How it works:
        - K seed vectors (learned) attend to T tokens
        - Each seed produces one output vector
        - Result: T tokens → K summary vectors
    
    Args:
        hidden_size: Embedding dimension (H)
        n_heads: Number of attention heads
        n_seeds: Number of output vectors (K)
        dropout: Dropout probability
    
    Input:
        X: [B, T, H] - token embeddings
        mask: [B, T] - optional, True for positions to ignore
    
    Output:
        [B, K, H] - K summary vectors
    
    Example:
        >>> pma = PMA(hidden_size=768, n_heads=8, n_seeds=4)
        >>> tokens = torch.randn(2, 512, 768)
        >>> pooled = pma(tokens)  # [2, 4, 768]
    """
    
    def __init__(self, hidden_size, n_heads, n_seeds=1, dropout=0.1):
        super().__init__()
        
        # Learned seed vectors
        # Each seed learns to extract different information
        self.seeds = nn.Parameter(torch.randn(n_seeds, hidden_size))
        
        # Single MAB for pooling
        self.mab = MAB(hidden_size, n_heads, dropout)
    
    def forward(self, X, mask=None):
        """
        Args:
            X: [B, T, H] - input tokens
            mask: [B, T] - True = ignore this token
        
        Returns:
            [B, K, H] - K summary vectors (K = n_seeds)
        """
        B = X.size(0)
        
        # Expand seeds for batch: [K, H] → [B, K, H]
        S = self.seeds.unsqueeze(0).expand(B, -1, -1)
        
        # Seeds attend to tokens
        # [B, K, H] attends to [B, T, H] → [B, K, H]
        return self.mab(Q=S, KV=X, mask=mask)