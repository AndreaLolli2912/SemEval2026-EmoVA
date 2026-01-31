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
        pma_num_seeds: Number of PMA output vectors per document
        isab_inducing_points: Number of ISAB inducing points (None to skip ISAB)
        n_heads: Number of attention heads for ISAB/PMA
        lstm_hidden_dim: LSTM hidden dimension
        lstm_num_layers: Number of LSTM layers
        lstm_bidirectional: Whether LSTM is lstm_bidirectional
        dropout: Dropout probability
        constrain_arousal: Whether to constrain arousal to [0, 2]
        verbose: Print shape information during forward pass
    """
    
    def __init__(
        self,
        # Encoder params
        model_path,
        encoder_bitfit=False,
        encoder_use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_bias = None,
        lora_dropout=0.1,
        # Set attention params
        pma_num_seeds=4,
        isab_inducing_points=32,
        n_heads=8,
        # LSTM params
        lstm_hidden_dim=256,
        lstm_num_layers=2,
        lstm_bidirectional=True,
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
        
        # 1. Transformer encoder (frozen)
        self.encoder = TransformerEncoder(
            model_path=model_path,
            fine_tune_bias=encoder_bitfit,
            use_lora=encoder_use_lora,     
            lora_r=lora_r,                 
            lora_alpha=lora_alpha, 
            lora_bias = lora_bias,
            lora_dropout=lora_dropout,     
            verbose=verbose
        )
        
        hidden_size = self.encoder.hidden_size
        
        # 2. ISAB (optional - skip if isab_inducing_points is None)
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
        if pma_num_seeds is not None and pma_num_seeds > 0:
            self.pma = PMA(
                hidden_size=hidden_size,
                n_heads=n_heads,
                n_seeds=pma_num_seeds,
                dropout=dropout,
                verbose=verbose
            )
            text_dim = hidden_size * pma_num_seeds
        else:
            self.pma = None
            text_dim = hidden_size
        
        # 4. LSTM encoder
        lstm_input_dim = text_dim
        
        self.lstm = LSTMEncoder(
            input_dim=lstm_input_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            bidirectional=lstm_bidirectional,
            dropout=dropout,
            verbose=verbose
        )
        
        # 5. Prediction head
        self.head = PredictionHead(
            input_dim=self.lstm.output_dim,
            dropout=dropout,
            constrain_output=constrain_output,
            verbose=verbose
        )
        
        if self.verbose:
            print(f"\n[AffectModel] Initialized")
            print(f"  Encoder: {model_path} (frozen)")
            print(f"  ISAB: {isab_inducing_points} inducing points" if isab_inducing_points else "  ISAB: disabled")
            print(f"  PMA: {pma_num_seeds} seeds")
            print(f"  LSTM: input={lstm_input_dim}, hidden={lstm_hidden_dim}, layers={lstm_num_layers}, bidir={lstm_bidirectional}")
            print(f"  Head: output=2, constrain_output={constrain_output}\n")

    
    def _mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask

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
            print(f"    seq_mask:       {seq_mask.shape} -> {mask.sum().item()} valid documents\n")
        
        # 1. Flatten valid documents
        input_ids_flat = input_ids[mask]          # [N_valid, T]
        attention_mask_flat = attention_mask[mask] # [N_valid, T]
        
        if self.verbose:
            print(f"\n  Step 1: Flatten valid documents")
            print(f"    input_ids_flat: {input_ids_flat.shape}\n")
        
        # 2. Encode with transformer
        tokens, padding_mask = self.encoder(input_ids_flat, attention_mask_flat)
        
        if self.verbose:
            print(f"\n  Step 2: Transformer encoding")
            print(f"    tokens: {tokens.shape}")
            print(f"    padding_mask: {padding_mask.shape}\n")
        
        # 3. ISAB (optional enrichment)
        if self.isab is not None:
            tokens = self.isab(tokens, padding_mask)
            if self.verbose:
                print(f"\n  Step 3: ISAB enrichment")
                print(f"    tokens (enriched): {tokens.shape}\n")
        
        # 4. PMA (pool to fixed size)
        if self.pma is not None:
            emb_flat = self.pma(tokens, padding_mask) # [N_valid, pma_num_seeds, H]
            emb_flat = emb_flat.view(emb_flat.size(0), -1)
        else:
            emb_flat = self._mean_pooling(tokens, attention_mask_flat)
        
        if self.verbose:
            print(f"\n  Step 4: PMA pooling")
            print(f"    emb_flat: {emb_flat.shape}\n")
        
        # 5. Reconstruct padded tensor
        '''emb = torch.zeros(
            B, S, *emb_flat.shape[1:],
            device=emb_flat.device, dtype=emb_flat.dtype
        )
        emb[mask] = emb_flat
        emb = emb.view(B, S, -1)  # [B, S, pma_num_seeds * H]'''

        emb = torch.zeros(
            B, S, emb_flat.size(-1), # Prende l'ultima dimensione automaticamente
            device=emb_flat.device, dtype=emb_flat.dtype
        )
        emb[mask] = emb_flat
        
        if self.verbose:
            print(f"\n  Step 5: Reconstruct for LSTM")
            print(f"    emb (reconstructed): {emb.shape}\n")
        
        # 6. LSTM
        lstm_out = self.lstm(emb, seq_lengths)  # [B, S, hidden*2]
        
        if self.verbose:
            print(f"\n  Step 6: LSTM")
            print(f"    lstm_out: {lstm_out.shape}\n")
        
        # 7. Predict
        predictions = self.head(lstm_out)  # [B, S, 2]
        
        if self.verbose:
            print(f"\n  Step 7: Prediction head")
            print(f"    predictions: {predictions.shape}\n")
        
        return predictions



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
        encoder_use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        # Set attention params
        pma_num_seeds=4,
        isab_inducing_points=32,
        n_heads=8,
        # LSTM params
        lstm_hidden_dim=256,
        lstm_num_layers=2,
        lstm_bidirectional=True,
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
            use_lora=encoder_use_lora,     
            lora_r=lora_r,                 
            lora_alpha=lora_alpha,         
            lora_dropout=lora_dropout,     
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
        if pma_num_seeds is not None and pma_num_seeds > 0:
            self.pma = PMA(
                hidden_size=hidden_size,
                n_heads=n_heads,
                n_seeds=pma_num_seeds,
                dropout=dropout,
                verbose=verbose
            )
            text_dim = hidden_size * pma_num_seeds
        else:
            self.pma = None
            text_dim = hidden_size
        
        # 4. LSTM encoder
        lstm_input_dim = text_dim + 2
        
        self.lstm = LSTMEncoder(
            input_dim=lstm_input_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            bidirectional=lstm_bidirectional,
            dropout=dropout,
            verbose=verbose
        )

        lstm_output_dim = self.lstm.output_dim

        head_dim =  lstm_output_dim + 2
        # 5. Prediction head
        self.head = nn.Sequential(
              nn.Dropout(dropout),
              nn.Linear(head_dim, head_dim // 2),
              nn.ReLU(),                                      
              nn.Linear(head_dim // 2, 2)                      
        )
        
        if self.verbose:
            print(f"\n[AffectModel2a] Initialized")
            print(f"  Encoder: {model_path}")
            print(f"  LSTM Input Dim: {lstm_input_dim} (Text + 2 history)")
    
    def _mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask
    def forward(self, input_ids, attention_mask, history_va, seq_lengths, seq_mask):
        """
        Args:
            input_ids: [B, S, T]
            attention_mask: [B, S, T]
            history_va: [B, S, 2]
            seq_lengths: [B]
            seq_mask: [B, S]
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
        if self.pma is not None:
            emb_flat = self.pma(tokens, padding_mask) # [N_valid, pma_num_seeds, H]
            emb_flat = emb_flat.view(emb_flat.size(0), -1)
        else:
            emb_flat = self._mean_pooling(tokens, attention_mask_flat)
        
        if self.verbose:
            print(f"\n  Step 4: PMA pooling")
            print(f"    emb_flat: {emb_flat.shape}\n")
        
        # 5. Reconstruct padded tensor
        emb = torch.zeros(
            B, S, emb_flat.size(-1), # Prende l'ultima dimensione automaticamente
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
        last_idx = (seq_lengths - 1).clamp(min=0) 
        
        last_hidden_state = lstm_out[batch_idx, last_idx, :]

        # retrieve the last known values
        last_known_value = history_va[batch_idx, last_idx, :] # [B, 2]
        
        # --- 8. Prediction Head ---
        head_input = torch.cat([last_hidden_state, last_known_value], dim=-1) # [B, Hidden + 2]
        predictions = self.head(head_input) # [B, 2]
        
        return predictions
