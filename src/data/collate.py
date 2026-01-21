"""Dataloader collate function entry point"""
import torch

def create_collate_fn(tokenizer_wrapper, pad_value=0.0):
    """
    Factory function to create a collate function with a specific tokenizer.
    
    Args:
        tokenizer_wrapper: TokenizerWrapper instance
        pad_value: Value to use for padding valence/arousal (default: 0.0)
    
    Returns:
        collate_fn: Function to use with DataLoader
    """
    
    def collate_fn(batch):
        """
        Collate function that:
        1. Tokenizes all texts in the batch
        2. Pads sequences to max length in batch
        3. Creates attention masks for sequences
        
        Args:
            batch: List of items from EmoVADataset
        
        Returns:
            Dictionary with batched and padded data ready for model
        """
        # Get max sequence length in this batch
        max_seq_len = max(item['seq_length'] for item in batch)
        batch_size = len(batch)
        
        # Initialize lists for batch data
        batch_user_ids = []
        batch_texts_tokenized = []
        batch_valences = []
        batch_arousals = []
        batch_seq_lengths = []
        batch_attention_masks = []
        batch_timestamps = []
        batch_text_ids = []
        
        for item in batch:
            seq_len = item['seq_length']
            pad_len = max_seq_len - seq_len
            
            # User ID
            batch_user_ids.append(item['user_id'])
            
            # Tokenize texts using TokenizerWrapper
            texts = item['texts']
            tokenized = tokenizer_wrapper(texts)
            # tokenized is a BatchEncoding with:
            #   - input_ids: [seq_len, max_text_length] <- now always max_text_length!
            #   - attention_mask: [seq_len, max_text_length]
            
            # Pad sequence dimension if needed
            if pad_len > 0:
                max_text_length = tokenizer_wrapper.max_len
                
                # Create padding tensors for sequences
                pad_input_ids = torch.zeros(pad_len, max_text_length, dtype=torch.long)
                pad_attention_mask = torch.zeros(pad_len, max_text_length, dtype=torch.long)
                
                # Concatenate real + padding
                tokenized_input_ids = torch.cat([tokenized['input_ids'], pad_input_ids], dim=0)
                tokenized_attention_mask = torch.cat([tokenized['attention_mask'], pad_attention_mask], dim=0)
            else:
                tokenized_input_ids = tokenized['input_ids']
                tokenized_attention_mask = tokenized['attention_mask']
            
            batch_texts_tokenized.append({
                'input_ids': tokenized_input_ids,
                'attention_mask': tokenized_attention_mask
            })
            
            # Pad valences and arousals
            padded_valences = torch.cat([
                item['valences'],
                torch.full((pad_len,), pad_value, dtype=torch.float32)
            ])
            padded_arousals = torch.cat([
                item['arousals'],
                torch.full((pad_len,), pad_value, dtype=torch.float32)
            ])
            
            batch_valences.append(padded_valences)
            batch_arousals.append(padded_arousals)
            
            # Create sequence attention mask: 1 for real, 0 for padding
            seq_attention_mask = torch.cat([
                torch.ones(seq_len, dtype=torch.float32),
                torch.zeros(pad_len, dtype=torch.float32)
            ])
            batch_attention_masks.append(seq_attention_mask)
            
            # Sequence length
            batch_seq_lengths.append(seq_len)
            
            # Timestamps and text_ids (keep as lists)
            batch_timestamps.append(item['timestamps'])
            batch_text_ids.append(item['text_ids'])
        
        # Stack tokenized texts
        input_ids = torch.stack([item['input_ids'] for item in batch_texts_tokenized])
        # Shape: [batch_size, max_seq_len, max_text_length]
        
        attention_mask_text = torch.stack([item['attention_mask'] for item in batch_texts_tokenized])
        # Shape: [batch_size, max_seq_len, max_text_length]
        
        # Create final batch dictionary
        batch_dict = {
            'user_ids': batch_user_ids,  # List of user IDs
            'input_ids': input_ids,  # [batch_size, max_seq_len, max_text_length]
            'attention_mask': attention_mask_text,  # [batch_size, max_seq_len, max_text_length]
            'valences': torch.stack(batch_valences),  # [batch_size, max_seq_len]
            'arousals': torch.stack(batch_arousals),  # [batch_size, max_seq_len]
            'seq_attention_mask': torch.stack(batch_attention_masks),  # [batch_size, max_seq_len]
            'seq_lengths': torch.tensor(batch_seq_lengths, dtype=torch.long),  # [batch_size]
            'timestamps': batch_timestamps,  # List of lists
            'text_ids': batch_text_ids  # List of lists
        }
        
        return batch_dict
    
    return collate_fn