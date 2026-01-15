# utils.py
import os
import random
from typing import Dict, List

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Sets the random seed for reproducibility[cite: 245]."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_vad_lexicon(path: str) -> Dict[str, List[float]]:
    """
    Parses NRC VAD Lexicon v2.
    Expected format: Word <tab> Valence <tab> Arousal <tab> Dominance
    Returns: Dict {word: [V, A, D]}
    """
    print(f"Loading Lexicon from {path}...")
    lexicon = {}
    with open(path, "r") as f:
        for line in f:
            word, v, a, d = line.strip().split("\t")
            lexicon[word] = [float(v), float(a), float(d)]
    return lexicon
