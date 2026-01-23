import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class TransformerEncoder(nn.Module):
    """
    Fixed transformer encoder. Config locked at init.
    
    Args:
        model_path: HuggingFace model path
        pooling: 'mean', 'max', 'cls', 'attention'
        n_groups: If provided, outputs [B, G, H] instead of [B, H]
        grouped_mode: 'split' or 'attention' (only used if n_groups is set)
        conv_kernel_size: For grouped attention mode
        freeze_backbone: Whether to freeze transformer weights
    """
    
    def __init__(
        self, 
        model_path, 
        pooling='mean',
        n_groups=None,
        grouped_mode='attention',
        conv_kernel_size=3,
        freeze_backbone=True
    ):
        super().__init__()
        
        # Store fixed config
        self.pooling = pooling
        self.n_groups = n_groups
        self.grouped_mode = grouped_mode if n_groups else None
        
        # Load backbone
        self.backbone = AutoModel.from_pretrained(model_path)
        self.hidden_size = self.backbone.config.hidden_size
        
        if freeze_backbone:
            self._freeze_backbone()
        
        # Build the ONE pooling head we need
        self._build_pooling_head(conv_kernel_size)
    
    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
    
    def _build_pooling_head(self, conv_kernel_size):
        """Initialize only the layers needed for this config."""
        
        if self.n_groups is not None:
            # Grouped output
            if self.grouped_mode == 'attention':
                self.conv = nn.Conv1d(
                    self.hidden_size, self.hidden_size,
                    kernel_size=conv_kernel_size,
                    padding=conv_kernel_size // 2
                )
                self.group_proj = nn.Linear(self.hidden_size, self.n_groups)
            # 'split' mode needs no extra params
        
        else:
            # Single output
            if self.pooling == 'attention':
                self.attn_proj = nn.Linear(self.hidden_size, 1)
            # 'mean', 'max', 'cls' need no extra params
    
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [B, T]
            attention_mask: [B, T]
        
        Returns:
            [B, H] if n_groups is None, else [B, G, H]
        """
        tokens = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state  # [B, T, H]
        print("tokens.shape", tokens.shape)
        if self.n_groups is not None:
            return self._pool_grouped(tokens, attention_mask)
        else:
            return self._pool_single(tokens, attention_mask)
    
    def _pool_single(self, tokens, mask):
        """[B, T, H] -> [B, H]"""
        if self.pooling == 'cls':
            return tokens[:, 0]
        
        elif self.pooling == 'mean':
            mask_exp = mask.unsqueeze(-1).float()
            return (tokens * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)
        
        elif self.pooling == 'max':
            tokens = tokens.masked_fill(~mask.unsqueeze(-1).bool(), -1e9)
            return tokens.max(dim=1).values
        
        elif self.pooling == 'attention':
            scores = self.attn_proj(tokens).squeeze(-1)  # [B, T]
            scores = scores.masked_fill(mask == 0, -1e9)
            weights = F.softmax(scores, dim=-1).unsqueeze(-1)
            return (tokens * weights).sum(dim=1)
    
    def _pool_grouped(self, tokens, mask):
        """[B, T, H] -> [B, G, H]"""
        if self.grouped_mode == 'split':
            return self._grouped_split(tokens, mask)
        elif self.grouped_mode == 'attention':
            return self._grouped_attention(tokens, mask)
    
    def _grouped_split(self, tokens, mask):
        B, T, H = tokens.shape
        G = self.n_groups
        
        # Pad to make divisible
        pad = (G - T % G) % G
        if pad > 0:
            tokens = F.pad(tokens, (0, 0, 0, pad))
            mask = F.pad(mask, (0, pad), value=0)
        
        # Reshape and mean pool each chunk
        tokens = tokens.view(B, G, -1, H)
        mask = mask.view(B, G, -1).unsqueeze(-1).float()
        
        return (tokens * mask).sum(2) / mask.sum(2).clamp(min=1e-9)
    
    def _grouped_attention(self, tokens, mask):
        B, T, H = tokens.shape
        
        # Local context
        x = self.conv(tokens.transpose(1, 2)).transpose(1, 2)  # [B, T, H]
        x = x * mask.unsqueeze(-1).float()
        
        # Group attention
        logits = self.group_proj(x)  # [B, T, G]
        logits = logits.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        
        weights = F.softmax(logits, dim=1).transpose(1, 2)  # [B, G, T]
        return torch.bmm(weights, x)  # [B, G, H]
    
    def encode(self, inputs):
        """Convenience method accepting dict."""
        return self.forward(inputs['input_ids'], inputs['attention_mask'])