import torch
import torch.nn as nn
from transformers import AutoModel

class TransformerEncoder(nn.Module):
    """
    Transformer encoder with optional BitFit (Bias-term Fine-tuning).
    
    Args:
        model_path: HuggingFace model path
        fine_tune_bias: If True, unfreezes bias terms (BitFit) while keeping weights frozen.
        verbose: If True, print shapes and flow information
    """
    
    def __init__(self, model_path, fine_tune_bias=False, verbose=False):
        super().__init__()
        
        self.verbose = verbose
        self.fine_tune_bias = fine_tune_bias
        self.backbone = AutoModel.from_pretrained(model_path)
        self.hidden_size = self.backbone.config.hidden_size
        
        self._configure_gradients()

        if self.verbose:
            print(f"[TransformerEncoder] Loaded: {model_path}")
            print(f"[TransformerEncoder] Hidden size: {self.hidden_size}")
            print(f"[TransformerEncoder] Backbone frozen: True\n")
    
    def _configure_gradients(self):
        for name, param in self.backbone.named_parameters():
            if self.fine_tune_bias and "bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.backbone.eval()

    def train(self, mode=True):
        """
        Override to manage backbone state.
        For BitFit, we usually keep the backbone structure (Dropout) in eval mode
        to prevent noise, even though we are updating bias params.
        """
        super().train(mode)
        self.backbone.eval()
        return self
    
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [B, T]
            attention_mask: [B, T] (1 = real, 0 = padding)
        
        Returns:
            tokens: [B, T, H] - token embeddings
            padding_mask: [B, T] - True = ignore (for PMA/ISAB)
        """
        if self.verbose:
            print(f"\n[TransformerEncoder] Forward pass")
            print(f"  Input:")
            print(f"    input_ids:      {input_ids.shape}")
            print(f"    attention_mask: {attention_mask.shape}")
            print(f"    real tokens:    {attention_mask.sum().item()} / {attention_mask.numel()}\n")
        
        tokens = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        
        padding_mask = (attention_mask == 0)
        
        if self.verbose:
            print(f"  Output:")
            print(f"    tokens:       {tokens.shape}")
            print(f"    padding_mask: {padding_mask.shape}")
            print(f"    positions to ignore: {padding_mask.sum().item()}\n")
        
        return tokens, padding_mask
    
    def encode(self, inputs):
        """Convenience method accepting dict."""
        return self.forward(inputs['input_ids'], inputs['attention_mask'])