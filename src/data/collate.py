import torch

def create_collate_fn(tokenizer_wrapper, pad_value=0.0):

    def collate_fn(batch):
        batch_size = len(batch)
        seq_lengths = [item['seq_length'] for item in batch]
        max_seq_len = max(seq_lengths)

        # Flatten texts for tokenization
        flat_texts = []
        text_counts = []  # how many texts per user

        for item in batch:
            texts = item['texts']
            flat_texts.extend(texts)
            text_counts.append(len(texts))

        # Single tokenizer call
        tokenized = tokenizer_wrapper(flat_texts)
        input_ids_flat = tokenized['input_ids']
        attention_mask_flat = tokenized['attention_mask']

        max_text_len = input_ids_flat.size(1)

        
        # Allocate padded tensors
        input_ids = torch.zeros(
            batch_size, max_seq_len, max_text_len, dtype=torch.long
        )
        attention_mask_text = torch.zeros_like(input_ids)

        valences = torch.full(
            (batch_size, max_seq_len), pad_value, dtype=torch.float32
        )
        arousals = torch.full(
            (batch_size, max_seq_len), pad_value, dtype=torch.float32
        )

        seq_attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)

        # Fill tensors
        cursor = 0
        for i, item in enumerate(batch):
            seq_len = item['seq_length']
            count = text_counts[i]

            input_ids[i, :seq_len] = input_ids_flat[cursor:cursor + count]
            attention_mask_text[i, :seq_len] = attention_mask_flat[cursor:cursor + count]

            valences[i, :seq_len] = item['valences']
            arousals[i, :seq_len] = item['arousals']

            seq_attention_mask[i, :seq_len] = True

            cursor += count

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
