import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class EmoVADataset(Dataset):
    """
    Dataset for longitudinal affect assessment.
    Each item is one user's complete temporal sequence.
    Select the history length of a user.
    """

    def __init__(self, path, dtype=torch.float32, constrain_output = False, max_history = 10):
        self.path = path
        self.dtype = dtype
        self.constrain_output = constrain_output
        self.max_history_length = max_history

        # Load dataset
        df = pd.read_csv(path)
      
        # Sort by datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

        target_col = ['state_change_valence', 'state_change_arousal']
        df = df.dropna(subset=target_cols)

        self.df = df

        self.samples = []

        # normalization 
        VALENCE_MAX = 2.0 # valence new range [-1,1]
        AROUSAL_MAX = 2.0 # arousal new range [0,1]
        
        for user_id, group in df.groupby('user_id', sort=False):
          text_ids = group['text_id'].tolist(),
          texts = group['text'].tolist(),
          timestamps = group['timestamp'].to_numpy(),
          collection_phases = group['collection_phase'].tolist(),
          is_words = group['is_words'].tolist(),
          raw_valence = group['valence'].to_numpy(dtype=np.float32)
          raw_arousal = group['arousal'].to_numpy(dtype=np.float32)

          target_valence = group['state_change_valence'].to_numpy(dtype=np.float32)
          target_arousal = group['state_change_arousal'].to_numpy(dtype=np.float32)

          num_texts = len(texts_ids)
          for i in range(num_texts):
            # define the sliding window from start to i
            slw_start = max(0, i-self.max_history_length+1)
            slw_end = i+1

            seq_text = text[slw_start:slw_end]
            seq_valence= raw_valence[slw_start:slw_end]
            seq_arousal=raw_arousal[[slw_start:slw_end]

            seq_target_valence=target_valence[slw_start:slw_end]
            seq_target_arousal=target_arousal[slw_start:slw_end]

            self.samples.append({
                'user_id': user_id,      
                'text_id': text_ids[i],   
                'texts': seq_texts,       
                'input_valences': seq_vals, 
                'input_arousals': seq_aros,
                'target': seq_target_valence, seq_target_arousal]
                })
    def __len__(self):
        return len(self.user_data)

    def __getitem__(self, idx):
        data = self.user_data[idx]
        return data
