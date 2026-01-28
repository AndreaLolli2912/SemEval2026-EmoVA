import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class EmoVADataset(Dataset):
    """
    Dataset for longitudinal affect assessment.
    Each item is one user's complete temporal sequence.
    """

    def __init__(self, path, dtype=torch.float32, max_seq_len=16, overlap=0) :
        self.path = path
        self.max_seq_len = max_seq_len
        self.overlap = overlap # useful for the context awareness
        self.dtype = dtype

        # Load dataset
        df = pd.read_csv(path)

        # Sort by datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

        self.df = df

        self.samples = []
        for user_id, group in df.groupby('user_id', sort=False):
            text_ids = group['text_id'].tolist()
            texts = group['text'].tolist(),
            timestamps = group['timestamp'].to_numpy(),
            collection_phases = group['collection_phase'].tolist(),
            is_words = group['is_words'].tolist(),
            valences = group['valence'].to_numpy(dtype=np.float32),
            arousals = group['arousal'].to_numpy(dtype=np.float32),

            total_text = len(texts)

            step = max_seq_len - overlap
            if step < 1: step = 1

            for i in range(0, total_text, step):
              # select the indices for beginning and endig slice
              end = min(i+max_seq_len, total_text)

              chunk = {
                    'user_id': user_id, # ID is the same for all the pieces
                    'chunk_id': f"{user_id}_{i}", # ID for this pieces
                    'text_ids': text_ids[i:end],
                    'texts': texts[i:end],
                    'timestamps': timestamps[i:end],
                    'collection_phases': collection_phases[i:end],
                    'is_words': is_words[i:end],
                    'valences': valences[i:end],
                    'arousals': arousals[i:end],
              }
              self.samples.append(chunk)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        texts = data['texts']
        valences = data['valences']
        arousals = data['arousals']
        
        return {
            'user_id': data['user_id'],
            'text_ids': data['text_ids'],
            'texts': texts, 
            'timestamps': data['timestamps'],
            'valences': torch.from_numpy(valences).to(self.dtype),
            'arousals': torch.from_numpy(arousals).to(self.dtype),
            'seq_length': len(texts)
        }
