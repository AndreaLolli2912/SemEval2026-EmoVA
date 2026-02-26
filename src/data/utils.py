"""Entru point for data utility functions"""

from torch.utils.data import DataLoader

from src.data.dataset import EmoVADataset, EmoVADataset2a
from src.data.collate import create_collate_fn, create_collate_fn_2a
from src.models.tokenizer_wrapper import TokenizerWrapper

def setup_dataloader(csv_path, tokenizer_path, max_text_length=128, 
                     batch_size=8, shuffle=True, num_workers=0, constrain_output=False):
    """
    Complete setup function that creates everything you need.
    
    Args:
        csv_path: Path to your CSV file
        tokenizer_path: HuggingFace model path for tokenizer
        max_text_length: Maximum length for each text
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data
        num_workers: Number of workers for DataLoader
        constrain_output: Normalization
    
    Returns:
        dataloader: Ready-to-use DataLoader
        dataset: The dataset instance
    """
    # 1. Create tokenizer wrapper
    tokenizer_wrapper = TokenizerWrapper(
        path=tokenizer_path,
        max_len=max_text_length
    )
    
    # 2. Create dataset
    dataset = EmoVADataset(csv_path,
                           constrain_output=constrain_output
                          )
    
    # 3. Create collate function
    collate_fn = create_collate_fn(
        tokenizer_wrapper=tokenizer_wrapper,
        pad_value=0.0
    )
    
    # 4. Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    return dataloader, dataset

def setup_dataloader_2a(csv_path, tokenizer_path, max_text_length=128, 
                     batch_size=8, shuffle=True, num_workers=0, constrain_output=False,  max_history=5, step=2):
    """
    Complete setup function that creates everything you need.
    
    Args:
        csv_path: Path to your CSV file
        tokenizer_path: HuggingFace model path for tokenizer
        max_text_length: Maximum length for each text
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data
        num_workers: Number of workers for DataLoader
        constrain_output: Normalization
        max_history: Number of text per user
        step: Step for the sliding window
    
    Returns:
        dataloader: Ready-to-use DataLoader
        dataset: The dataset instance
    """
    # 1. Create tokenizer wrapper
    tokenizer_wrapper = TokenizerWrapper(
        path=tokenizer_path,
        max_len=max_text_length
    )
    
    # 2. Create dataset
    dataset = EmoVADataset2a(csv_path,
                             constrain_output=constrain_output,
                             max_history=max_history,
                             step=step
                             )
    
    # 3. Create collate function
    collate_fn = create_collate_fn_2a(
        tokenizer_wrapper=tokenizer_wrapper,
        pad_value=0.0
    )
    
    # 4. Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    return dataloader, dataset
