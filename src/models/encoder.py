import torch
import torch.nn as nn
from transformers import AutoModel


class TransformerEncoder(nn.Module):
    """
    Frozen transformer encoder.
    
    Returns raw token embeddings for downstream pooling (PMA/ISAB).
    No trainable parameters - backbone is always frozen.
    
    Args:
        model_path: HuggingFace model path
    
    Input:
        input_ids: [B, T]
        attention_mask: [B, T]
    
    Output:
        tokens: [B, T, H]
        padding_mask: [B, T] (True = ignore, for use with PMA/ISAB)
    """
    
    def __init__(self, model_path):
        super().__init__()
        
        self.backbone = AutoModel.from_pretrained(model_path)
        self.hidden_size = self.backbone.config.hidden_size
        
        self._freeze_backbone()
    
    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
    
    def train(self, mode=True):
        """Override to keep backbone in eval mode."""
        super().train(mode)
        self.backbone.eval()
        return self
    
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [B, T]
            attention_mask: [B, T] (1 = real, 0 = padding)
        
        Returns:
            tokens: [B, T, H] - token embeddings (includes CLS at position 0)
            padding_mask: [B, T] - True = ignore (for PMA/ISAB)
        """
        with torch.no_grad():
            tokens = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state
        
        padding_mask = (attention_mask == 0)
        
        return tokens, padding_mask
    
    def encode(self, inputs):
        """Convenience method accepting dict."""
        return self.forward(inputs['input_ids'], inputs['attention_mask'])