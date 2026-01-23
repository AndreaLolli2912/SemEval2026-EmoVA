import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class EmoVADataset(Dataset):
    """
    Dataset for longitudinal affect assessment.
    Each item is one user's complete temporal sequence.
    """

    def __init__(self, path, dtype=torch.float32):
        self.path = path
        self.dtype = dtype

        # Load dataset
        df = pd.read_csv(path)

        # Sort by datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

        self.df = df

        # Precompute user -> index ranges
        self.user_groups = []
        for user_id, group in df.groupby('user_id', sort=False):
            self.user_groups.append({
                'user_id': user_id,
                'indices': group.index.to_numpy(),
                'seq_length': len(group)
            })

    def __len__(self):
        return len(self.user_groups)

    def __getitem__(self, idx):
        meta = self.user_groups[idx]
        idxs = meta['indices']
        df = self.df.loc[idxs]

        return {
            'user_id': meta['user_id'],
            'text_ids': df['text_id'].tolist(),
            'texts': df['text'].tolist(),
            'timestamps': df['timestamp'].to_numpy(),
            'collection_phases': df['collection_phase'].tolist(),
            'is_words': df['is_words'].tolist(),
            'valences': torch.from_numpy(
                df['valence'].to_numpy(dtype=np.float32)
            ).to(self.dtype),
            'arousals': torch.from_numpy(
                df['arousal'].to_numpy(dtype=np.float32)
            ).to(self.dtype),
            'seq_length': meta['seq_length']
        }