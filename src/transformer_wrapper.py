import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

class TransformerWrapper(nn.Module):
    def __init__(self, tokenizer_path, model_path, device=None, 
                 use_conv_grouped_pooling=False, n_groups=None, max_seq_length=512, 
                 use_learnable_single_pooling=False, conv_kernel_size=3):
        """
        Args:
            tokenizer_path: Hugging Face tokenizer path
            model_path: Hugging Face model path
            device: 'cuda' or 'cpu'
            use_conv_grouped_pooling: if True, uses Conv1D + Adaptive Pooling for grouped encoding.
            n_groups: number of groups for grouped encoding (required if use_conv_grouped_pooling=True).
            max_seq_length: maximum sequence length for tokenization.
            use_learnable_single_pooling: if True, initializes weights for attention-based single pooling.
            conv_kernel_size: kernel size for the depthwise convolution used in grouped pooling.
        """
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load HuggingFace components
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)
        # Default to eval mode for the base HF model so dropout is off by default
        self.model.eval() 
        
        self.hidden_size = self.model.config.hidden_size
        self.max_seq_length = max_seq_length
        self.n_groups = n_groups
        
        # Configuration flags
        self.use_conv_grouped_pooling = use_conv_grouped_pooling
        self.use_learnable_single_pooling = use_learnable_single_pooling

        # --- Initialize Learnable Parameters ---

        # 1. Parameters for single learnable attention pooling
        if self.use_learnable_single_pooling:
            # A learnable query vector to attend to the sequence
            self.attention_query = nn.Parameter(torch.randn(1, 1, self.hidden_size))
            self.attention_proj = nn.Linear(self.hidden_size, self.hidden_size)

        # 2. Parameters for grouped convolution pooling
        if self.use_conv_grouped_pooling:
            if self.n_groups is None:
                raise ValueError("n_groups must be specified when use_conv_grouped_pooling=True")
            
            # Use a fixed-size depthwise convolution to capture local context.
            # Adaptive pooling will handle varying sequence lengths later.
            # padding=conv_kernel_size//2 maintains sequence length ("same" padding behavior roughly)
            self.conv1d_grouped = nn.Conv1d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=conv_kernel_size,
                stride=1,
                padding=conv_kernel_size // 2, 
                groups=self.hidden_size,  # Depthwise: independent per embedding dimension
                bias=True
            )

    def set_training_mode(self, mode='pooling_only'):
        """
        Helper to set training states for fine-tuning.
        """
        if mode == 'pooling_only':
            self.model.eval() # Freeze HF Transformer backbone
            self.train()      # Unfreeze wrappers internal parameters (pooling weights)
            # Explicitly turn off grads for base model weights
            for param in self.model.parameters():
                param.requires_grad = False
        elif mode == 'full':
            self.model.train() # Unfreeze everything, enable dropout in HF model
            self.train()
            for param in self.model.parameters():
                param.requires_grad = True
        elif mode == 'inference':
            self.model.eval()
            self.eval()
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _get_token_embeddings(self, texts, max_length=None):
        """Helper to run tokenizer and base transformer."""
        max_len = max_length or self.max_seq_length
        if isinstance(texts, str):
            texts = [texts]
            
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=max_len
        ).to(self.device)

        # The model's current training state (set by set_training_mode) determines gradient flow.
        outputs = self.model(**inputs)
        
        token_embeddings = outputs.last_hidden_state # (B, T, H)
        attention_mask = inputs["attention_mask"]    # (B, T)
        return token_embeddings, attention_mask

    # -----------------------------
    # Single Embedding Methods
    # -----------------------------
    def encode(self, texts, max_length=None, pooling="mean"):
        """
        Returns single embedding per text: (B, H)
        Pooling options: 'mean', 'max', 'cls', 'learnable_attention'
        """
        token_embeddings, attention_mask = self._get_token_embeddings(texts, max_length)

        if pooling == "learnable_attention":
            if not self.use_learnable_single_pooling:
                 raise ValueError("Initialize with use_learnable_single_pooling=True to use this method.")
            return self._learnable_attention_pooling(token_embeddings, attention_mask)
        
        elif pooling == "mean":
            return self._mean_pooling(token_embeddings, attention_mask)
        elif pooling == "max":
            return self._max_pooling(token_embeddings, attention_mask)
        elif pooling == "cls":
            return token_embeddings[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

    def _learnable_attention_pooling(self, token_embeddings, attention_mask):
        """Attention-based pooling inspired by BERT's pooler but with learnable query."""
        # Project token embeddings: (B, T, H) -> (B, T, H)
        projected = self.attention_proj(token_embeddings)
        
        # Compute attention scores using learnable query: (B, T, H) x (1, H, 1) -> (B, T, 1)
        scores = torch.matmul(projected, self.attention_query.transpose(1, 2))
        scores = scores.squeeze(-1) # (B, T)
        
        # Mask padding tokens before softmax
        mask_value = -1e9 if scores.dtype == torch.float32 else -1e4
        scores = scores.masked_fill(attention_mask == 0, mask_value)
        
        # Softmax to get weights: (B, T) -> (B, T, 1)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        
        # Weighted sum: (B, H)
        pooled = torch.sum(token_embeddings * weights, dim=1)
        return pooled

    # -----------------------------
    # Grouped Embedding Methods
    # -----------------------------
    def encode_grouped(self, texts, n_groups=None, pooling="mean"):
        """
        Returns embeddings: (B, n_groups, H)
        If use_conv_grouped_pooling=True in init, uses Conv1D+AdaptiveAvgPool (recommended).
        Otherwise, splits sequence into chunks and applies standard pooling.
        """
        target_n_groups = n_groups or self.n_groups
        if target_n_groups is None:
             raise ValueError("n_groups must be provided either in init or call.")

        token_embeddings, attention_mask = self._get_token_embeddings(texts)
        
        if self.use_conv_grouped_pooling:
            return self._encode_grouped_conv(token_embeddings, attention_mask, target_n_groups)
        else:
            return self._encode_grouped_split(token_embeddings, attention_mask, target_n_groups, pooling)

    def _encode_grouped_conv(self, token_embeddings, attention_mask, n_groups):
        """
        Robust implementation of convolution pooling for varying sequence lengths.
        Uses fixed Conv1d -> Re-masking -> Adaptive Pooling -> Normalization.
        """
        B, T, H = token_embeddings.shape
        
        # 1. Prepare masks
        mask_expanded = attention_mask.unsqueeze(-1).float() # (B, T, 1)
        mask_channels_last = mask_expanded.transpose(1, 2)   # (B, 1, T)
        
        # 2. Initial masking of input embeddings
        masked_embeddings = token_embeddings * mask_expanded
        
        # 3. Apply Conv1d (B, Channels, Length)
        x = masked_embeddings.transpose(1, 2) # (B, H, T)
        features = self.conv1d_grouped(x)     # (B, H, T) due to padding
        
        # 4. Re-masking (CRITICAL FIX 1)
        # Ensure padding positions are zeroed out again after convolution
        features = features * mask_channels_last

        # 5. Adaptive Pooling
        # (B, H, T) -> (B, H, n_groups)
        # This averages whatever is in the window, including zeros from padding.
        pooled_sum = F.adaptive_avg_pool1d(features, output_size=n_groups)
        
        # 6. Normalization (CRITICAL FIX 2)
        # Calculate the density of valid tokens in each adaptive pool window.
        # If a window covers 50% valid tokens, the mask density will be 0.5.
        mask_density = F.adaptive_avg_pool1d(mask_channels_last, output_size=n_groups) # (B, 1, n_groups)
        
        # Divide by density to get the true average of valid tokens.
        # Clamp to avoid division by zero for completely empty segments.
        pooled_normalized = pooled_sum / torch.clamp(mask_density, min=1e-9)
        
        # 7. Return to (B, n_groups, H)
        return pooled_normalized.transpose(1, 2)

    def _encode_grouped_split(self, token_embeddings, attention_mask, n_groups, pooling):
        """Splits sequence into roughly equal chunks and applies mean/max pooling."""
        B, T, H = token_embeddings.shape
        
        # Calculate split points
        base_size = T // n_groups
        remainder = T % n_groups
        sizes = [base_size + (1 if i < remainder else 0) for i in range(n_groups)]
        
        grouped_embeddings = []
        start_idx = 0
        for size in sizes:
            end_idx = start_idx + size
            if size == 0: 
                # Handle edge case where T < n_groups. 
                # Must append a tensor of shape (B, 1, H) to cat later.
                grouped_embeddings.append(torch.zeros(B, 1, H, device=self.device))
                continue
                
            group_tokens = token_embeddings[:, start_idx:end_idx, :]
            group_mask = attention_mask[:, start_idx:end_idx]
            
            if pooling == "mean":
                pooled = self._mean_pooling(group_tokens, group_mask) # (B, H)
            elif pooling == "max":
                pooled = self._max_pooling(group_tokens, group_mask)  # (B, H)
            else:
                raise ValueError(f"Unknown pooling: {pooling}")
                
            grouped_embeddings.append(pooled.unsqueeze(1)) # (B, 1, H)
            start_idx = end_idx
            
        return torch.cat(grouped_embeddings, dim=1) # (B, n_groups, H)

    # -----------------------------
    # Static Pooling Helpers
    # -----------------------------
    @staticmethod
    def _mean_pooling(token_embeddings, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        # Clamp to avoid division by zero for completely padded sequences
        counts = torch.clamp(mask.sum(dim=1), min=1e-9) 
        return summed / counts
    
    @staticmethod
    def _max_pooling(token_embeddings, attention_mask):
        mask = attention_mask.unsqueeze(-1).bool()
        # Fill padding with very small number before max
        masked_tokens = token_embeddings.masked_fill(~mask, -1e9)
        return torch.max(masked_tokens, dim=1).values