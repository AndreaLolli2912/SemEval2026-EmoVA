import torch

def create_collate_fn(tokenizer_wrapper, pad_value = -100.0, max_length = 128):

    def collate_fn(batch):
        batch_size = len(batch)
        seq_lengths = [item['seq_length'] for item in batch]
        max_seq_len = max(seq_lengths)

        # Flatten texts for tokenization
        all_text = []
        all_valence = []
        all_arousal = []
        seq_lenght = []

        for item in batch:
            texts = item['texts']
            all_text.extend(texts)
            seq_lengths.append(len(texts))
        
            all_valences.append(item['valences'])
            all_arousals.append(item['arousals'])

        # Single tokenizer call
        tokenized = tokenizer_wrapper(all_text, paddig = True, truncation = True, max_length= max_length, return_tensors='pt')
        # label padding for diffent users' number of text
        valence_padded = pad_sequence(all_valences, batch_first=True, padding_value= pad_value)
        arousals_padded = pad_sequence(all_arousals, batch_first=True, padding_value= pad_value)

        seq_attention_mask = torch.zeros(batch_size, max(seq_lengths), dtype=torch.bool)

        # fill tensor
        for i, length in enumerate(seq_len):
            seq_attention_mask[i, :length] = 1

        # Assemble batch
        return {
            'user_ids': [item['user_id'] for item in batch],
            'text_ids': [item['text_ids'] for item in batch],
            'input_ids': input_ids,  # [B, T, L]
            'attention_mask': attention_mask_text,  # [B, T, L]
            'timestamps': [item['timestamps'] for item in batch],
            'collection_phases': [item['collection_phases'] for item in batch],
            'is_words': [item['is_words'] for item in batch],
            'valences': valences,  # [B, T]
            'arousals': arousals,  # [B, T]
            'seq_attention_mask': seq_attention_mask,  # [B, T]
            'seq_lengths': torch.tensor(seq_lengths, dtype=torch.long),   
        }

    return collate_fn
