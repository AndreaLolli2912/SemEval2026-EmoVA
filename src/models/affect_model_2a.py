import torch
from torch import nn

from src.models.encoder import TransformerEncoder
from src.models.set_attention import ISAB, PMA
from src.models.lstm import LSTMEncoder

class AffectModel2a(nn.Module):
    """
    Forecasting model: Transformer -> ISAB -> PMA -> Fusion -> LSTM -> Prediction Head
    Input: Sequence of posts + Sequence of Valence/Arousal history
    Output: Single prediction [B, 2] (Delta Valence, Delta Arousal)
    """
    
    def __init__(
        self,
        # Encoder params
        model_path,
        encoder_bitfit=False,
        # Set attention params
        pma_num_seeds=4,
        isab_inducing_points=32,
        n_heads=8,
        # LSTM params
        lstm_hidden_dim=256,
        lstm_num_layers=2,
        lstm_bidirectional=False,
        # Shared params
        dropout=0.3,
        # Head params
        constrain_output=False,
        # Debug
        verbose=False
    ):
        super().__init__()
        
        self.verbose = verbose
        self.pma_num_seeds = pma_num_seeds
        self.isab_inducing_points = isab_inducing_points
        
        # 1. Transformer encoder (frozen/bitfit)
        self.encoder = TransformerEncoder(
            model_path=model_path,
            fine_tune_bias=encoder_bitfit,
            verbose=verbose
        )
        
        hidden_size = self.encoder.hidden_size
        
        # 2. ISAB (optional)
        if isab_inducing_points is not None:
            self.isab = ISAB(
                hidden_size=hidden_size,
                n_heads=n_heads,
                n_inducing=isab_inducing_points,
                dropout=dropout,
                verbose=verbose
            )
        else:
            self.isab = None
        
        # 3. PMA (pooling)
        self.pma = PMA(
            hidden_size=hidden_size,
            n_heads=n_heads,
            n_seeds=pma_num_seeds,
            dropout=dropout,
            verbose=verbose
        )
        
        # 4. LSTM encoder
        lstm_input_dim = (hidden_size * pma_num_seeds) + 2
        
        self.lstm = LSTMEncoder(
            input_dim=lstm_input_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            bidirectional=lstm_bidirectional,
            dropout=dropout,
            verbose=verbose
        )

        lstm_output_dim = self.lstm.output_dim

        # 5. Prediction head
        self.head = nn.Sequential(
              nn.Dropout(dropout),
              nn.Linear(lstm_output_dim, lstm_output_dim // 2),
              nn.ReLU(),                                      
              nn.Linear(lstm_output_dim // 2, 2)                      
        )
        
        if self.verbose:
            print(f"\n[AffectModel2a] Initialized")
            print(f"  Encoder: {model_path}")
            print(f"  LSTM Input Dim: {lstm_input_dim} (Text + 2 history)")

    def forward(self, input_ids, attention_mask, history_va, seq_lengths, seq_mask):
        """
        Args:
            input_ids: [B, S, T]
            attention_mask: [B, S, T]
            history_va: [B, S, 2] <--- AGGIUNTO QUESTO
            seq_lengths: [B]
            seq_mask: [B, S] (Serve per appiattire il batch per il Transformer)
        """
        B, S, T = input_ids.shape
        mask = seq_mask.bool()
        
        # --- 1. Flatten valid documents (Batch * Seq -> N_valid) ---
        input_ids_flat = input_ids[mask]          
        attention_mask_flat = attention_mask[mask] 
        
        # --- 2. Encode with Transformer ---
        tokens, padding_mask = self.encoder(input_ids_flat, attention_mask_flat)
        
        # --- 3. ISAB (Optional) ---
        if self.isab is not None:
            tokens = self.isab(tokens, padding_mask)
        
        # --- 4. PMA Pooling ---
        emb_flat = self.pma(tokens, padding_mask)  # [N_valid, pma_num_seeds, H]
        
        # --- 5. Reconstruct padded tensor (N_valid -> Batch * Seq) ---
        emb = torch.zeros(
            B, S, *emb_flat.shape[1:],
            device=emb_flat.device, dtype=emb_flat.dtype
        )

        emb[mask] = emb_flat
        
        # Flattening dei seed PMA: [B, S, pma_num_seeds * H]
        emb = emb.view(B, S, -1) 
        
        # --- 6. Fusion (Text + History) & LSTM ---
        # emb: [B, S, TextDim]
        # history_va: [B, S, 2]
        lstm_input = torch.cat([emb, history_va], dim=-1) # [B, S, TextDim + 2]
        
        lstm_out = self.lstm(lstm_input, seq_lengths)  # [B, S, LSTM_Hidden]
        
        # --- 7. Select Last Valid State (Forecasting) ---
        batch_idx = torch.arange(B, device=lstm_out.device)
        # Usiamo clamp(min=0) per sicurezza
        last_idx = (seq_lengths - 1).clamp(min=0) 
        
        last_hidden_state = lstm_out[batch_idx, last_idx, :]
        
        # --- 8. Prediction Head ---
        predictions = self.head(last_hidden_state) # [B, 2]
        
        return predictions
