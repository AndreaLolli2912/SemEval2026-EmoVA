import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

class TransformerWrapper(nn.Module):
    def __init__(self, tokenizer_path, model_path, device=None, use_trainable_pooling=False, 
                 use_conv_pooling=False, n_groups=None, max_seq_length=512, 
                 overlap_pooling=False):
        """
        Args:
            tokenizer_path: Hugging Face tokenizer
            model_path: Hugging Face model
            device: 'cuda' or 'cpu'
            use_trainable_pooling: whether to use trainable weighted pooling
            use_conv_pooling: if True, uses Conv1D for grouped pooling
            n_groups: number of groups (required if use_conv_pooling=True, fixed per experiment)
            max_seq_length: maximum sequence length
            overlap_pooling: if True, use overlapping windows (stride = kernel_size // 2)
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # default inference mode
        
        # Get hidden size from model config
        self.hidden_size = self.model.config.hidden_size
        
        # Pooling options
        self.use_trainable_pooling = use_trainable_pooling
        self.use_conv_pooling = use_conv_pooling
        self.n_groups = n_groups
        self.max_seq_length = max_seq_length
        self.overlap_pooling = overlap_pooling
        
        # Validate conv pooling config
        if use_conv_pooling and n_groups is None:
            raise ValueError("n_groups must be specified when use_conv_pooling=True")
        
        # Initialize trainable pooling weights (lazy init)
        if use_trainable_pooling:
            self.pool_weights = None
        
        # For conv pooling, we'll use dynamic depthwise conv weights
        # This allows handling variable sequence lengths correctly
        # Depthwise: each channel (embedding dimension) pools independently
        if use_conv_pooling:
            # Store learnable depthwise conv weights
            # Shape: (hidden_size, 1, kernel_size) for depthwise conv
            self.conv_weights = None
            self.conv_bias = None
        else:
            self.conv_weights = None
            self.conv_bias = None

    def set_training_mode(self, mode='pooling_only'):
        """
        Set training mode for different components.
        
        Args:
            mode: 'pooling_only' (default) - only train pooling layers
                  'full' - train transformer + pooling (not recommended for large models)
                  'inference' - all in eval mode
        """
        if mode == 'pooling_only':
            self.model.eval()  # Freeze transformer
            self.train()  # Enable training for pooling layers
        elif mode == 'full':
            self.model.train()  # Unfreeze transformer
            self.train()
        elif mode == 'inference':
            self.model.eval()
            self.eval()
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'pooling_only', 'full', or 'inference'")
    
    # ----------------------------- 
    # Standard single embedding
    # ----------------------------- 
    def encode(self, texts, max_length=512, pooling="mean"):
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        token_embeddings = outputs.last_hidden_state  # (B, T, H)
        attention_mask = inputs["attention_mask"]  # (B, T)
        
        if pooling == "mean":
            return self._mean_pooling(token_embeddings, attention_mask)
        elif pooling == "max":
            return self._max_pooling(token_embeddings, attention_mask)
        elif pooling == "cls":
            return token_embeddings[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

    # ----------------------------- 
    # Grouped embeddings (trainable or conv)
    # ----------------------------- 
    def encode_grouped(self, texts, n_groups=None, pooling="mean", training=False):
        """
        Returns embeddings: (B, n_groups, H)
        Supports:
        - standard trainable weighted pooling
        - Conv1D grouped pooling (if use_conv_pooling=True, mask-aware)
        
        Args:
            texts: input text(s)
            n_groups: number of groups (if None, uses self.n_groups from __init__)
            pooling: pooling method ('mean' or 'max')
            training: whether in training mode (for trainable pooling)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Use n_groups from init if not provided
        if n_groups is None:
            n_groups = self.n_groups
        
        if n_groups is None:
            raise ValueError("n_groups must be specified either in __init__ or in encode_grouped()")
        
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length
        ).to(self.device)
        
        # Always disable gradients for the transformer model
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        token_embeddings = outputs.last_hidden_state  # (B, T, H)
        attention_mask = inputs["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
        
        B, T, H = token_embeddings.shape
        
        # Enable gradients for pooling operations if in training mode
        if training:
            token_embeddings = token_embeddings.detach().requires_grad_(True)
        
        # ----------------------------- 
        # Mask-aware depthwise Conv1D pooling (dynamic)
        # ----------------------------- 
        if self.use_conv_pooling:
            # Calculate padding if needed to make T divisible by n_groups
            remainder = T % n_groups
            if remainder > 0:
                pad_total = n_groups - remainder
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                
                # Pad token embeddings and mask (already on correct device)
                token_embeddings = F.pad(token_embeddings, (0, 0, pad_left, pad_right), value=0)
                attention_mask = F.pad(attention_mask, (0, 0, pad_left, pad_right), value=0)
                
                T = token_embeddings.size(1)  # Update T after padding
            
            # Calculate kernel and stride based on actual sequence length
            kernel_size = T // n_groups
            stride = kernel_size // 2 if self.overlap_pooling else kernel_size
            
            # Initialize or verify depthwise conv weights match current kernel size
            if self.conv_weights is None or self.conv_weights.size(2) != kernel_size:
                # Initialize learnable depthwise conv weights for this kernel size
                # Shape: (hidden_size, 1, kernel_size) for depthwise convolution
                # Each output channel only looks at its corresponding input channel
                self.conv_weights = nn.Parameter(
                    torch.randn(H, 1, kernel_size, device=self.device) / (kernel_size ** 0.5)
                )
                self.conv_bias = nn.Parameter(torch.zeros(H, device=self.device))
                # Register parameters properly
                self.register_parameter('_conv_weights', self.conv_weights)
                self.register_parameter('_conv_bias', self.conv_bias)
            
            # Zero out padding tokens before convolution
            token_embeddings = token_embeddings * attention_mask
            
            # Apply dynamic depthwise Conv1D using F.conv1d
            # groups=H means each channel is convolved independently
            x = token_embeddings.transpose(1, 2)  # (B, H, T)
            conv_out = F.conv1d(
                x, 
                weight=self.conv_weights, 
                bias=self.conv_bias,
                stride=stride,
                padding=0,
                groups=H  # Depthwise: each embedding dimension pools independently
            )  # (B, H, n_groups) or (B, H, 2*n_groups-1) if overlapping
            
            conv_out = conv_out.transpose(1, 2)  # (B, n_groups, H) or (B, 2*n_groups-1, H)
            
            # Compute mask sum for each window to normalize properly
            mask_transposed = attention_mask.transpose(1, 2)  # (B, 1, T)
            mask_sum = F.avg_pool1d(
                mask_transposed, 
                kernel_size=kernel_size, 
                stride=stride
            ) * kernel_size  # (B, 1, n_groups_actual) - sum of valid tokens per window
            
            # Normalize by actual number of valid tokens in each window
            mask_sum = mask_sum.transpose(1, 2)  # (B, n_groups_actual, 1)
            conv_out = conv_out / torch.clamp(mask_sum, min=1e-9)
            
            # If overlapping, we may have more outputs than n_groups
            # Apply adaptive pooling to get exactly n_groups
            if conv_out.size(1) != n_groups:
                # (B, n_groups_actual, H) -> (B, H, n_groups_actual) -> (B, H, n_groups)
                conv_out = F.adaptive_avg_pool1d(
                    conv_out.transpose(1, 2), 
                    n_groups
                ).transpose(1, 2)
            
            return conv_out  # (B, n_groups, H)
        
        # ----------------------------- 
        # Original grouped pooling with even splits
        # ----------------------------- 
        # Create more even groups
        base_group_size = T // n_groups
        remainder = T % n_groups
        
        # Distribute remainder across first groups
        group_sizes = [base_group_size + (1 if i < remainder else 0) for i in range(n_groups)]
        
        starts = []
        ends = []
        cumsum = 0
        for size in group_sizes:
            starts.append(cumsum)
            cumsum += size
            ends.append(cumsum)
        
        grouped_embeddings = []
        
        for start, end in zip(starts, ends):
            group_tokens = token_embeddings[:, start:end, :]  # (B, group_len, H)
            group_mask = attention_mask[:, start:end, :]  # (B, group_len, 1)
            
            # Trainable weighted pooling
            if self.use_trainable_pooling:
                group_len = group_tokens.size(1)
                
                # Lazy initialization of pool weights
                if self.pool_weights is None or self.pool_weights.size(1) != group_len:
                    # Initialize as parameter with proper device placement
                    self.pool_weights = nn.Parameter(
                        torch.ones(1, group_len, 1, device=self.device, dtype=token_embeddings.dtype)
                    )
                    # Register it properly
                    self.register_parameter('_pool_weights', self.pool_weights)
                
                weights = torch.softmax(self.pool_weights, dim=1)
                pooled = torch.sum(group_tokens * weights * group_mask, dim=1) / torch.clamp(
                    group_mask.sum(dim=1), min=1e-9
                )
            else:
                # Standard pooling
                if pooling == "mean":
                    pooled = torch.sum(group_tokens * group_mask, dim=1) / torch.clamp(
                        group_mask.sum(dim=1), min=1e-9
                    )
                elif pooling == "max":
                    masked_tokens = group_tokens.masked_fill(group_mask == 0, -1e9)
                    pooled = torch.max(masked_tokens, dim=1).values
                else:
                    raise ValueError(f"Unknown pooling method: {pooling}")
            
            grouped_embeddings.append(pooled.unsqueeze(1))
        
        return torch.cat(grouped_embeddings, dim=1)  # (B, n_groups, H)

    # ----------------------------- 
    # Pooling helpers
    # ----------------------------- 
    @staticmethod
    def _mean_pooling(token_embeddings, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts
    
    @staticmethod
    def _max_pooling(token_embeddings, attention_mask):
        mask = attention_mask.unsqueeze(-1).bool()
        masked_tokens = token_embeddings.masked_fill(~mask, -1e9)
        return torch.max(masked_tokens, dim=1).values