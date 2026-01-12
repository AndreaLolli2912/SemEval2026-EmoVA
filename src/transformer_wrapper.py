import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

class TransformerWrapper(nn.Module):
    def __init__(self, tokenizer_path, model_path, device=None, 
                 use_attention_grouped_pooling=False, n_groups=None, max_seq_length=512, 
                 use_learnable_single_pooling=False, conv_kernel_size=3):
        """
        Improved transformer wrapper with cleaner attention-based grouped pooling.
        
        Args:
            tokenizer_path: Hugging Face tokenizer path
            model_path: Hugging Face model path
            device: 'cuda' or 'cpu'
            use_attention_grouped_pooling: if True, uses attention-based grouped encoding
            n_groups: number of groups for grouped encoding (required if use_attention_grouped_pooling=True)
            max_seq_length: maximum sequence length for tokenization
            use_learnable_single_pooling: if True, initializes weights for attention-based single pooling
            conv_kernel_size: kernel size for the convolution used in grouped pooling
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
        self.use_attention_grouped_pooling = use_attention_grouped_pooling
        self.use_learnable_single_pooling = use_learnable_single_pooling

        # --- Initialize Learnable Parameters ---
        
        # 1. Parameters for single learnable attention pooling
        if self.use_learnable_single_pooling:
            # A learnable query vector to attend to the sequence
            self.single_projection = nn.Linear(self.hidden_size, 1)

        # 2. Parameters for attention-based grouped pooling
        if self.use_attention_grouped_pooling:
            if self.n_groups is None:
                raise ValueError("n_groups must be specified when use_attention_grouped_pooling=True")
            
            # Local context via convolution (optional but recommended)
            self.local_conv = nn.Conv1d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=conv_kernel_size,
                stride=1,  # Keep sequence length
                padding=conv_kernel_size // 2,
                bias=True
            )
            
            # Learnable projection to create group-specific attention scores
            # Each group learns which positions in the sequence to attend to
            self.group_projection = nn.Linear(self.hidden_size, self.n_groups)
            
            print(f"Initialized attention-based grouped pooling: {self.n_groups} groups")

    def set_training_mode(self, mode='pooling_only'):
        """
        Helper to set training states for fine-tuning.
        
        Args:
            mode: 'pooling_only', 'full', or 'inference'
        """
        if mode == 'pooling_only':
            self.model.eval()  # Freeze HF Transformer backbone
            self.train()       # Unfreeze wrapper's internal parameters (pooling weights)
            # Explicitly turn off grads for base model weights
            for param in self.model.parameters():
                param.requires_grad = False
        elif mode == 'full':
            self.model.train()  # Unfreeze everything, enable dropout in HF model
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

        # The model's current training state (set by set_training_mode) determines gradient flow
        outputs = self.model(**inputs)
        
        token_embeddings = outputs.last_hidden_state  # (B, T, H)
        attention_mask = inputs["attention_mask"]     # (B, T)
        return token_embeddings, attention_mask

    # -----------------------------
    # Single Embedding Methods
    # -----------------------------
    def encode(self, texts, max_length=None, pooling="mean"):
        """
        Returns single embedding per text: (B, H)
        
        Args:
            texts: List of strings or single string
            max_length: Maximum sequence length (default: self.max_seq_length)
            pooling: One of 'mean', 'max', 'cls', 'learnable_attention'
        
        Returns:
            embeddings: (B, H) tensor
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
        """ Learnable attention-based pooling."""
        # Project to scalar attention scores: (B, T, H) -> (B, T, 1) -> (B, T)
        scores = self.single_projection(token_embeddings).squeeze(-1)
        
        # Mask padding tokens before softmax
        mask_value = -1e9 if scores.dtype == torch.float32 else -1e4
        scores = scores.masked_fill(attention_mask == 0, mask_value)
        
        # Softmax to get weights: (B, T) -> (B, T, 1)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        
        # Weighted sum: (B, T, H) * (B, T, 1) -> sum over T -> (B, H)
        pooled = torch.sum(token_embeddings * weights, dim=1)
        return pooled

    # -----------------------------
    # Grouped Embedding Methods
    # -----------------------------
    def encode_grouped(self, texts, n_groups=None, pooling="mean"):
        """
        Returns grouped embeddings: (B, n_groups, H)
        
        Args:
            texts: List of strings or single string
            n_groups: Number of groups (default: self.n_groups)
            pooling: For split-based: 'mean' or 'max'. Ignored if use_attention_grouped_pooling=True
        
        Returns:
            grouped_embeddings: (B, n_groups, H) tensor
        """
        target_n_groups = n_groups or self.n_groups
        if target_n_groups is None:
            raise ValueError("n_groups must be provided either in init or call.")

        token_embeddings, attention_mask = self._get_token_embeddings(texts)
        
        if self.use_attention_grouped_pooling:
            return self._encode_grouped_attention(token_embeddings, attention_mask, target_n_groups)
        else:
            return self._encode_grouped_split(token_embeddings, attention_mask, target_n_groups, pooling)

    def _encode_grouped_attention(self, token_embeddings, attention_mask, n_groups):
        """
        Attention-based grouped pooling: each group learns what to attend to.
        
        Mathematical flow:
        1. Apply local convolution for context: (B, T, H) -> (B, T, H)
        2. Project to group attention scores: (B, T, H) -> (B, T, G)
        3. Mask padding and apply softmax per group
        4. Weighted aggregation: (B, G, T) @ (B, T, H) -> (B, G, H)
        """
        B, T, H = token_embeddings.shape
        
        # Step 1: Apply local convolution for context
        # Transpose for Conv1d: (B, T, H) -> (B, H, T)
        x = token_embeddings.transpose(1, 2)
        context_features = self.local_conv(x)  # (B, H, T)
        context_features = context_features.transpose(1, 2)  # (B, T, H)
        
        # Step 2: Mask padding in context features
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
        context_features = context_features * mask_expanded
        
        # Step 3: Compute attention logits for each group
        # (B, T, H) @ (H, G) -> (B, T, G)
        attention_logits = self.group_projection(context_features)
        
        # Step 4: Mask padding positions before softmax
        # We want softmax to ignore padding tokens
        mask_value = -1e9 if attention_logits.dtype == torch.float32 else -1e4
        # Expand mask: (B, T) -> (B, T, 1) -> (B, T, G)
        mask_for_logits = attention_mask.unsqueeze(-1).expand(-1, -1, n_groups)
        attention_logits = attention_logits.masked_fill(mask_for_logits == 0, mask_value)
        
        # Step 5: Softmax per group across sequence dimension
        # Each group gets a distribution over the sequence
        # (B, T, G) -> softmax over T dimension
        attention_weights = F.softmax(attention_logits, dim=1)  # (B, T, G)
        
        # Step 6: Transpose for batch matrix multiplication
        # (B, T, G) -> (B, G, T)
        attention_weights = attention_weights.transpose(1, 2)
        
        # Step 7: Weighted aggregation
        # (B, G, T) @ (B, T, H) -> (B, G, H)
        grouped_embeddings = torch.bmm(attention_weights, context_features)
        
        return grouped_embeddings

    def _encode_grouped_split(self, token_embeddings, attention_mask, n_groups, pooling):
        """
        Simple split-based grouping: divides sequence into chunks and pools each.
        This is a fallback method when attention-based grouping is not enabled.
        """
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
                # Handle edge case where T < n_groups
                grouped_embeddings.append(torch.zeros(B, 1, H, device=self.device))
                continue
                
            group_tokens = token_embeddings[:, start_idx:end_idx, :]
            group_mask = attention_mask[:, start_idx:end_idx]
            
            if pooling == "mean":
                pooled = self._mean_pooling(group_tokens, group_mask)  # (B, H)
            elif pooling == "max":
                pooled = self._max_pooling(group_tokens, group_mask)   # (B, H)
            else:
                raise ValueError(f"Unknown pooling: {pooling}")
                
            grouped_embeddings.append(pooled.unsqueeze(1))  # (B, 1, H)
            start_idx = end_idx
            
        return torch.cat(grouped_embeddings, dim=1)  # (B, n_groups, H)

    # -----------------------------
    # Static Pooling Helpers
    # -----------------------------
    @staticmethod
    def _mean_pooling(token_embeddings, attention_mask):
        """
        Mean pooling with proper masking of padding tokens.
        
        Args:
            token_embeddings: (B, T, H)
            attention_mask: (B, T)
        
        Returns:
            pooled: (B, H)
        """
        mask = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        # Clamp to avoid division by zero for completely padded sequences
        counts = torch.clamp(mask.sum(dim=1), min=1e-9) 
        return summed / counts
    
    @staticmethod
    def _max_pooling(token_embeddings, attention_mask):
        """
        Max pooling with proper masking of padding tokens.
        
        Args:
            token_embeddings: (B, T, H)
            attention_mask: (B, T)
        
        Returns:
            pooled: (B, H)
        """
        mask = attention_mask.unsqueeze(-1).bool()
        # Fill padding with very small number before max
        masked_tokens = token_embeddings.masked_fill(~mask, -1e9)
        return torch.max(masked_tokens, dim=1).values