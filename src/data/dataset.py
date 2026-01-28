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

        self.user_data = []

        # normalization [0,1] for arousal
        VALENCE_RANGE = 4.0
        VALENCE_MIN = -2.0
        VALENCE_MAX = 2.0
        AROUSAL_MAX = 2.0
        for user_id, group in df.groupby('user_id', sort=False):
            raw_valence = group['valence'].to_numpy(dtype=np.float32)
            raw_arousal = group['arousal'].to_numpy(dtype=np.float32)
            
            self.user_data.append({
                'user_id': user_id,
                'text_ids': group['text_id'].tolist(),
                'texts': group['text'].tolist(),
                'timestamps': group['timestamp'].to_numpy(),
                'collection_phases': group['collection_phase'].tolist(),
                'is_words': group['is_words'].tolist(),
                'valences': (raw_valence-VALENCE_MIN)/VALENCE_RANGE,
                'arousals': raw_arousal/AROUSAL_MAX,
            })

    def __len__(self):
        return len(self.user_data)

    def __getitem__(self, idx):
        data = self.user_data[idx]
        return {
            **data,
            'valences': torch.from_numpy(data['valences']).to(self.dtype),
            'arousals': torch.from_numpy(data['arousals']).to(self.dtype),
            'seq_length': len(data['texts'])
        }
