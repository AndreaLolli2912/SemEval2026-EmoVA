import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from src.training.utils import EarlyStopping

def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    config,
    clipper=None,
    collect_preds=False,
    max_collect_samples=5000,
):
    model.train()
    
    total_loss = 0.0
    total_samples = 0
    
    # Inizializza liste per raccolta predizioni (se richiesto)
    all_preds = [] if collect_preds else None
    all_targets = [] if collect_preds else None
    collected = 0

    criterion = nn.MSELoss()
    
    # Inizializza lo Scaler per Mixed Precision
    # Nota: se config.device è cpu, questo scaler non fa nulla (ed è ok)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(dataloader, desc="Training Task 2a", leave=False)
    
    for step, batch in enumerate(pbar):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        history_va = batch['history_va'].to(device, non_blocking=True)
        seq_lengths = batch['seq_lengths'].to(device, non_blocking=True)
        targets = batch['targets'].to(device, non_blocking=True)
        
        # Mixed Precision Context
        with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
            # Forward
            predictions = model(input_ids, attention_mask, history_va, seq_lengths)
            
            # Loss Calculation
            current_loss_raw = criterion(predictions, targets)
            
            # Normalize for accumulation
            current_loss = current_loss_raw / config.accumulation_steps
        
        # Backward
        scaler.scale(current_loss).backward()
        
        # Valore loss reale per il logging
        current_loss_val = current_loss_raw.detach().float().item()
        
        # Gradient accumulation
        if (step + 1) % config.accumulation_steps == 0:
            if clipper:
                scaler.unscale_(optimizer)
                clipper(model)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # Tracking
        with torch.no_grad():
            batch_size = input_ids.size(0)
            total_loss += (current_loss_val * batch_size)
            total_samples += batch_size
            
            if collect_preds and collected < max_collect_samples:
                preds_cpu = predictions.detach().cpu()
                targs_cpu = targets.detach().cpu()
                
                remaining = max_collect_samples - collected
                all_preds.append(preds_cpu[:remaining])
                all_targets.append(targs_cpu[:remaining])
                collected += preds_cpu.size(0)
        
        pbar.set_postfix({'mse': current_loss_val})
        
        # Cleanup
        del predictions, targets, input_ids, attention_mask, seq_lengths, history_va
    
    # Handle remaining gradients
    if (step + 1) % config.accumulation_steps != 0:
        if clipper:
            scaler.unscale_(optimizer)
            clipper(model)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    
    torch.cuda.empty_cache()
    
    result = {
        'loss': total_loss / max(total_samples, 1),
        'grad_norm': 0.0 # Semplificato per evitare errori se clipper non ha grad_norms
    }
    
    if collect_preds:
        result['preds'] = torch.cat(all_preds, dim=0)
        result['targets'] = torch.cat(all_targets, dim=0)
    
    return result

@torch.no_grad()
def eval_epoch(
    model,
    dataloader,
    device,
    config, 
    collect_preds=False, 
):
    """
    Evaluates model for Task 2A (Forecasting).
    Accumulates predictions to compute MSE and Pearson Correlation.
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0
    criterion = nn.MSELoss()

    all_preds = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc="Evaluating Task 2a", leave=False)
    
    for batch in pbar:
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        seq_lengths = batch['seq_lengths'].to(device, non_blocking
