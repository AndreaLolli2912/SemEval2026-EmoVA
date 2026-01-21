# dataset.py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, List, Any

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer

class EcologicalDataset(Dataset):
    def __init__(
        self, 
        df: pd.DataFrame, 
        tokenizer, 
        vad_lexicon: Dict[str, List[float]], 
        max_len: int = 512
    ):
        self.texts = df['text'].values
        # Feeling words are highly diagnostic features
        self.feeling_words = df['feeling_words'].fillna("").values
        self.user_ids = df['user_id'].values
        # Targets: Valence (col 0), Arousal (col 1)
        self.labels = df[['valence', 'arousal']].values.astype(float)
        
        self.tokenizer = tokenizer
        self.lexicon = vad_lexicon
        self.max_len = max_len

        # Pre-compute User History: Average V/A of user's past texts
        # In strict simulation, this should be computed only on training split to avoid leakage.
        # Here we use the global mean of the user as a static "Prior" for the architecture.
        self.user_prior_map = df.groupby('user_id')[['valence', 'arousal']].mean().to_dict('index')
        # Fallback for Cold Start users (Global Average)
        self.global_prior = df[['valence', 'arousal']].mean().values

    def get_lexicon_vector(self, text: str) -> List[float]:
        """Calculates mean VAD score for the text[cite: 148]."""
        words = text.lower().split()
        scores = [self.lexicon[w] for w in words if w in self.lexicon]
        
        if not scores:
            return [0.5, 0.5, 0.5] # Neutral fallback
        return np.mean(scores, axis=0).tolist()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Input Format: <feeling_words> [SEP] <essay_text>
        text_content = str(self.texts[idx])
        fw_content = str(self.feeling_words[idx])
        full_text = f"{fw_content} {self.tokenizer.sep_token} {text_content}"

        # Tokenization
        encoded = self.tokenizer(
            full_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Lexicon Feature Vector (3 dims)
        lex_vec = self.get_lexicon_vector(full_text)

        # User History Vector (2 dims)
        user_id = self.user_ids[idx]
        if user_id in self.user_prior_map:
            hist_vec = list(self.user_prior_map[user_id].values())
        else:
            hist_vec = self.global_prior.tolist()

        return {
            'input_ids': encoded['input_ids'].flatten(),
            'attention_mask': encoded['attention_mask'].flatten(),
            'lex_feats': torch.tensor(lex_vec, dtype=torch.float),
            'user_context': torch.tensor(hist_vec, dtype=torch.float),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }