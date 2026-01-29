import torch

def create_collate_fn(tokenizer, max_length=128):

    def collate_fn(batch):
        batch_size = len(batch)
        seq_lengths = [item['seq_length'] for item in batch]
        max_seq_len = max(seq_lengths)

        flat_texts = []
        text_counts = [] 

        for item in batch:
            texts = item['texts']
            flat_texts.extend(texts)
            text_counts.append(len(texts))

        # Tokenizzazione con parametri espliciti
        tokenized = tokenizer(
            flat_texts
        )
        
        input_ids_flat = tokenized['input_ids']
        attention_mask_flat = tokenized['attention_mask']
        
        max_text_len = input_ids_flat.size(1)
        input_ids = torch.zeros(batch_size, max_seq_len, max_text_len, dtype=torch.long)
        attention_mask_text = torch.zeros_like(input_ids)

        # Paddinf Value -100 
        valences = torch.full((batch_size, max_seq_len), -100.0, dtype=torch.float32)
        arousals = torch.full((batch_size, max_seq_len), -100.0, dtype=torch.float32)
        
        seq_attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)

        cursor = 0
        for i, item in enumerate(batch):
            seq_len = item['seq_length']
            count = text_counts[i]

            # Copiamo i dati flat nella struttura 3D
            input_ids[i, :seq_len] = input_ids_flat[cursor:cursor + count]
            attention_mask_text[i, :seq_len] = attention_mask_flat[cursor:cursor + count]

            # Copiamo le label
            valences[i, :seq_len] = item['valences']
            arousals[i, :seq_len] = item['arousals']

            seq_attention_mask[i, :seq_len] = True

            cursor += count

        return {
            'user_ids': [item['user_id'] for item in batch],
            'input_ids': input_ids,
            'attention_mask': attention_mask_text,
            'valences': valences,
            'arousals': arousals,
            'seq_attention_mask': seq_attention_mask,
            'seq_lengths': torch.tensor(seq_lengths, dtype=torch.long),
            # Metadata extra se servono
            'text_ids': [item['text_ids'] for item in batch],
        }

    return collate_fn
