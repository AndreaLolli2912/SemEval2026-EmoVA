import torch
import tqdm
from src.training.losses import masked_mse_loss

def train_epoch(model, dataloader, optimizer, clipper, device, accumulation_steps=1):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_samples = 0
    all_preds = []
    all_targets = []
    all_masks = []
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for step, batch in enumerate(pbar):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        seq_lengths = batch['seq_lengths'].to(device)
        seq_mask = batch['seq_attention_mask'].to(device)
        
        # Stack targets
        valences = batch['valences'].to(device)
        arousals = batch['arousals'].to(device)
        targets = torch.stack([valences, arousals], dim=-1)  # [B, S, 2]
        
        # Forward pass
        predictions = model(input_ids, attention_mask, seq_lengths, seq_mask)
        
        # Compute loss
        loss = masked_mse_loss(predictions, targets, seq_mask)
        loss = loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (step + 1) % accumulation_steps == 0:
            clipper(model)
            optimizer.step()
            optimizer.zero_grad()
        
        # Accumulate metrics
        total_loss += loss.item() * accumulation_steps * seq_mask.sum().item()
        total_samples += seq_mask.sum().item()
        
        # Store for CCC computation
        all_preds.append(predictions.detach())
        all_targets.append(targets.detach())
        all_masks.append(seq_mask.detach())
        
        pbar.set_postfix({'loss': loss.item() * accumulation_steps})
    
    # Handle remaining gradients
    if (step + 1) % accumulation_steps != 0:
        clipper(model)
        optimizer.step()
        optimizer.zero_grad()
    
    # Compute epoch metrics
    avg_loss = total_loss / total_samples
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    ccc_v, ccc_a = concordance_correlation_coefficient(all_preds, all_targets, all_masks)
    
    return {
        'loss': avg_loss,
        'ccc_valence': ccc_v.item(),
        'ccc_arousal': ccc_a.item(),
        'grad_norm': np.mean(clipper.grad_norms[-len(dataloader):])
    }