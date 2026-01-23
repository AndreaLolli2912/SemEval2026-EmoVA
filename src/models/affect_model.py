import torch
from torch import nn

from src.models.encoder import TransformerEncoder
from src.models.lstm import LSTMEncoder
from src.models.heads import PredictionHead


class AffectModel(nn.Module):
    """
    Full model: Transformer -> LSTM -> Prediction Head
    
    Input: tokenized texts [B, S, T]
    Output: [B, S, 2] (valence, arousal per timestep)
    """
    
    def __init__(
        self,
        # Encoder params
        model_path,
        n_groups=4,
        grouped_mode='attention',
        pooling='mean',
        freeze_backbone=True,
        conv_kernel_size=3,
        # LSTM params
        lstm_hidden=256,
        lstm_layers=2,
        bidirectional=True,
        # Shared params
        dropout=0.3,
        # Head params
        constrain_arousal=False
    ):
        super().__init__()
        
        self.n_groups = n_groups
        
        # 1. Transformer encoder
        self.encoder = TransformerEncoder(
            model_path=model_path,
            pooling=pooling,
            n_groups=n_groups,
            grouped_mode=grouped_mode,
            conv_kernel_size=conv_kernel_size,
            freeze_backbone=freeze_backbone
        )
        
        # 2. LSTM encoder
        lstm_input_dim = self.encoder.hidden_size * (n_groups if n_groups else 1)
        
        self.lstm = LSTMEncoder(
            input_dim=lstm_input_dim,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            cls_dim=self.encoder.hidden_size
        )
        
        # 3. Prediction head
        self.head = PredictionHead(
            input_dim=self.lstm.output_dim,
            dropout=dropout,
            constrain_arousal=constrain_arousal
        )
    
    def forward(self, input_ids, attention_mask, seq_lengths, seq_mask):
        """
        Args:
            input_ids: [B, S, T]
            attention_mask: [B, S, T]
            seq_lengths: [B]
            seq_mask: [B, S]
        
        Returns:
            predictions: [B, S, 2]
        """
        B, S, T = input_ids.shape
        mask = seq_mask.bool()
        
        # 1. Encode valid texts only
        print("input_ids.shape", input_ids.shape)
        emb_flat = self.encoder(input_ids[mask], attention_mask[mask])
        
        # 2. Reconstruct padded tensor
        emb = torch.zeros(
            B, S, *emb_flat.shape[1:],
            device=emb_flat.device, dtype=emb_flat.dtype
        )
        emb[mask] = emb_flat
        emb = emb.view(B, S, -1)  # [B, S, G*H] or [B, S, H]
        
        # 3. LSTM
        lstm_out = self.lstm(emb, seq_lengths)  # [B, S, hidden*2]
        
        # 4. Predict
        return self.head(lstm_out)  # [B, S, 2]


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