import torch
from torch import nn

from src.models.encoder import TransformerEncoder
from src.models.set_attention import ISAB, PMA
from src.models.lstm import LSTMEncoder
from src.models.heads import PredictionHead


class AffectModel(nn.Module):
    """
    Full model: Transformer -> ISAB -> PMA -> LSTM -> Prediction Head
    
    Input: tokenized texts [B, S, T]
    Output: [B, S, 2] (valence, arousal per timestep)
    
    Args:
        model_path: HuggingFace model path
        n_seeds: Number of PMA output vectors per document
        n_inducing: Number of ISAB inducing points (None to skip ISAB)
        n_heads: Number of attention heads for ISAB/PMA
        lstm_hidden: LSTM hidden dimension
        lstm_layers: Number of LSTM layers
        bidirectional: Whether LSTM is bidirectional
        dropout: Dropout probability
        constrain_arousal: Whether to constrain arousal to [0, 2]
        verbose: Print shape information during forward pass
    """
    
    def __init__(
        self,
        # Encoder params
        model_path,
        # Set attention params
        n_seeds=4,
        n_inducing=32,
        n_heads=8,
        # LSTM params
        lstm_hidden=256,
        lstm_layers=2,
        bidirectional=True,
        # Shared params
        dropout=0.3,
        # Head params
        constrain_arousal=False,
        # Debug
        verbose=False
    ):
        super().__init__()
        
        self.verbose = verbose
        self.n_seeds = n_seeds
        self.n_inducing = n_inducing
        
        # 1. Transformer encoder (frozen)
        self.encoder = TransformerEncoder(
            model_path=model_path,
            verbose=verbose
        )
        
        hidden_size = self.encoder.hidden_size
        
        # 2. ISAB (optional - skip if n_inducing is None)
        if n_inducing is not None:
            self.isab = ISAB(
                hidden_size=hidden_size,
                n_heads=n_heads,
                n_inducing=n_inducing,
                dropout=dropout,
                verbose=verbose
            )
        else:
            self.isab = None
        
        # 3. PMA (pooling)
        self.pma = PMA(
            hidden_size=hidden_size,
            n_heads=n_heads,
            n_seeds=n_seeds,
            dropout=dropout,
            verbose=verbose
        )
        
        # 4. LSTM encoder
        lstm_input_dim = hidden_size * n_seeds
        
        self.lstm = LSTMEncoder(
            input_dim=lstm_input_dim,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            verbose=verbose
        )
        
        # 5. Prediction head
        self.head = PredictionHead(
            input_dim=self.lstm.output_dim,
            dropout=dropout,
            constrain_arousal=constrain_arousal
        )
        
        if self.verbose:
            print(f"\n[AffectModel] Initialized")
            print(f"  Encoder: {model_path} (frozen)")
            print(f"  ISAB: {n_inducing} inducing points" if n_inducing else "  ISAB: disabled")
            print(f"  PMA: {n_seeds} seeds")
            print(f"  LSTM: input={lstm_input_dim}, hidden={lstm_hidden}, layers={lstm_layers}, bidir={bidirectional}")
            print(f"  Head: output=2, constrain_arousal={constrain_arousal}")
    
    def forward(self, input_ids, attention_mask, seq_lengths, seq_mask):
        """
        Args:
            input_ids: [B, S, T] - tokenized documents
            attention_mask: [B, S, T] - token-level mask
            seq_lengths: [B] - number of documents per user
            seq_mask: [B, S] - document-level mask
        
        Returns:
            predictions: [B, S, 2] - (valence, arousal) per document
        """
        B, S, T = input_ids.shape
        mask = seq_mask.bool()
        
        if self.verbose:
            print(f"\n[AffectModel] Forward pass")
            print(f"  Input shapes:")
            print(f"    input_ids:      {input_ids.shape} (B={B}, S={S}, T={T})")
            print(f"    attention_mask: {attention_mask.shape}")
            print(f"    seq_lengths:    {seq_lengths.shape} -> {seq_lengths.tolist()}")
            print(f"    seq_mask:       {seq_mask.shape} -> {mask.sum().item()} valid documents")
        
        # 1. Flatten valid documents
        input_ids_flat = input_ids[mask]          # [N_valid, T]
        attention_mask_flat = attention_mask[mask] # [N_valid, T]
        
        if self.verbose:
            print(f"\n  Step 1: Flatten valid documents")
            print(f"    input_ids_flat: {input_ids_flat.shape}")
        
        # 2. Encode with transformer
        tokens, padding_mask = self.encoder(input_ids_flat, attention_mask_flat)
        
        if self.verbose:
            print(f"\n  Step 2: Transformer encoding")
            print(f"    tokens: {tokens.shape}")
            print(f"    padding_mask: {padding_mask.shape}")
        
        # 3. ISAB (optional enrichment)
        if self.isab is not None:
            tokens = self.isab(tokens, padding_mask)
            if self.verbose:
                print(f"\n  Step 3: ISAB enrichment")
                print(f"    tokens (enriched): {tokens.shape}")
        
        # 4. PMA (pool to fixed size)
        emb_flat = self.pma(tokens, padding_mask)  # [N_valid, n_seeds, H]
        
        if self.verbose:
            print(f"\n  Step 4: PMA pooling")
            print(f"    emb_flat: {emb_flat.shape}")
        
        # 5. Reconstruct padded tensor
        emb = torch.zeros(
            B, S, *emb_flat.shape[1:],
            device=emb_flat.device, dtype=emb_flat.dtype
        )
        emb[mask] = emb_flat
        emb = emb.view(B, S, -1)  # [B, S, n_seeds * H]
        
        if self.verbose:
            print(f"\n  Step 5: Reconstruct for LSTM")
            print(f"    emb (reconstructed): {emb.shape}")
        
        # 6. LSTM
        lstm_out = self.lstm(emb, seq_lengths)  # [B, S, hidden*2]
        
        if self.verbose:
            print(f"\n  Step 6: LSTM")
            print(f"    lstm_out: {lstm_out.shape}")
        
        # 7. Predict
        predictions = self.head(lstm_out)  # [B, S, 2]
        
        if self.verbose:
            print(f"\n  Step 7: Prediction head")
            print(f"    predictions: {predictions.shape}")
        
        return predictions


def masked_mse_loss(pred, target, mask):
    """
    MSE loss only on valid timesteps.
    
    Args:
        pred: [B, S, 2]
        target: [B, S, 2]
        mask: [B, S] boolean
    
    Returns:
        scalar loss
    """
    mse = (pred - target) ** 2
    mse = mse * mask.unsqueeze(-1)
    return mse.sum() / (mask.sum() * pred.size(-1))