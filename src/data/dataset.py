"""Dataset Class entry point"""
import numpy  as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class EmoVADataset(Dataset):
    """
    Dataset for longitudinal affect assessment.
    Each item is one user's complete temporal sequence.
    """
    def __init__(self, path):
        """
        Args:
            path: Path to CSV file
        """
        self.path = path
        self.df = pd.read_csv(path)
        self.sequences = self._preprocess_dataset()

    def _preprocess_dataset(self):
        """
        Organize dataset into temporal sequences per user.
        """
        # Convert timestamp to datetime
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Sort by user_id and timestamp
        self.df = self.df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        
        sequences = []
        
        # Group by user_id
        for user_id, group in self.df.groupby('user_id'):
            # Sort by timestamp
            group = group.sort_values('timestamp').reset_index(drop=True)
            
            # Extract sequences
            texts = group['text'].tolist()
            valences = group['valence'].tolist()
            arousals = group['arousal'].tolist()
            timestamps = group['timestamp'].tolist()
            text_ids = group['text_id'].tolist()
            collection_phases = group['collection_phase'].tolist()
            
            # Create sequence dictionary
            seq_dict = {
                'user_id': user_id,
                'texts': texts,
                'valences': np.array(valences, dtype=np.float32),
                'arousals': np.array(arousals, dtype=np.float32),
                'timestamps': timestamps,
                'text_ids': text_ids,
                'collection_phases': collection_phases,
                'seq_length': len(texts)
            }
            
            sequences.append(seq_dict)
        
        return sequences

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Returns a single user's temporal sequence.
        Note: Text tokenization happens in collate_fn, not here!
        """
        seq = self.sequences[idx]
        
        item = {
            'user_id': seq['user_id'],
            'texts': seq['texts'],  # Raw text strings
            'valences': torch.from_numpy(seq['valences']),
            'arousals': torch.from_numpy(seq['arousals']),
            'timestamps': seq['timestamps'],
            'text_ids': seq['text_ids'],
            'collection_phases': seq['collection_phases'],
            'seq_length': seq['seq_length']
        }

        return item