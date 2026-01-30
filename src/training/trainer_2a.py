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
    
    # Inizializza liste per raccolta predizioni
    all_preds = [] if collect_preds else None
    all_targets = [] if collect_preds else None
    collected = 0

    criterion = nn.MSELoss()
    
    # Scaler per Mixed Precision
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(dataloader, desc="Training Task 2a", leave=False)
    
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        history_va = batch['history_va'].to(device, non_blocking=True)
        seq_lengths = batch['seq_lengths'].to(device, non_blocking=True)
        seq_mask = batch['seq_attention_mask'].to(device, non_blocking=True)
        targets = batch['targets'].to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
            predictions = model(input_ids, attention_mask, history_va, seq_lengths, seq_mask)
            current_loss_raw = criterion(predictions, targets)
            current_loss = current_loss_raw / config.accumulation_steps
        
        scaler.scale(current_loss).backward()
        current_loss_val = current_loss_raw.detach().float().item()
        
        if (step + 1) % config.accumulation_steps == 0:
            if clipper:
                scaler.unscale_(optimizer)
                clipper(model)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
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
        
        del predictions, targets, input_ids, attention_mask, seq_lengths, history_va
    
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
        'grad_norm': 0.0
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
    model.eval()
    total_loss = 0.0
    total_samples = 0
    criterion = nn.MSELoss()

    all_preds = {}
    all_targets = {}
    
    pbar = tqdm(dataloader, desc="Evaluating Task 2a", leave=False)
    
    for batch in pbar:
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        seq_lengths = batch['seq_lengths'].to(device, non_blocking=True)
        seq_mask = batch['seq_attention_mask'].to(device, non_blocking=True)
        history_va = batch['history_va'].to(device, non_blocking=True)
        targets = batch['targets'].to(device, non_blocking=True)

        user_ids = batch['user_ids']

        predictions = model(input_ids, attention_mask, history_va, seq_lengths, seq_mask)
        loss = criterion(predictions, targets)

        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        preds_cpu = predictions.cpu().numpy()
        targets_cpu = targets.cpu().numpy()

        for i, uid in enumerate(user_ids):
            if uid not in all_preds:
                all_preds[uid] = []
                all_targets[uid] = []
            all_preds[uid].append(preds_cpu[i])
            all_targets[uid].append(targets_cpu[i])
        
        pbar.set_postfix({'loss': loss.item()})
        del predictions, targets, input_ids, attention_mask, seq_lengths, history_va

    torch.cuda.empty_cache()

    final_preds_dict = {u: np.array(v) for u, v in all_preds.items()}
    final_gold_dict = {u: np.array(v) for u, v in all_targets.items()}
    
    result = {
        'loss': total_loss / max(total_samples, 1),
        'metrics': metrics,
        'score': metrics['overall/score'] 
     }
    
    return result

def train(model, train_loader, val_loader, loss_fn_name, optimizer, scheduler, device, config, clipper=None, save_dir='outputs_task2a'):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    n_epochs = config.epochs
    history = {'train_loss': [], 'val_loss': [], 'val_score': []}
    
    best_val_score = -1.0 
    best_epoch = 0
    best_checkpoint = None
    
    early_stopping = EarlyStopping(patience=config.patience, mode='max')
    
    for epoch in range(config.epochs):
        train_result = train_epoch(model, train_loader, optimizer, device, config, clipper)
        val_result = eval_epoch(model, val_loader, device, config)
        
        train_loss = train_result['loss']
        val_loss = val_result['loss']
        val_score = val_result['score']
        metrics = val_result['metrics']
        
        if scheduler:
            scheduler.step(val_loss)
        
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        print("-" * 30)
        print(f"  Train Loss:   {train_loss:.4f}")
        print(f"  Val Loss:     {val_loss:.4f}")
        print(f"  Val Score:    {val_score:.4f} (Avg Pearson r)")
        print(f"  > Valence r:  {metrics['valence/corr']:.4f}")
        print(f"  > Arousal r:  {metrics['arousal/corr']:.4f}")
        print(f"  LR: {optimizer.param_groups[-1]['lr']:.2e}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_score'].append(val_score)
        
        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch + 1
            
            saved_config = vars(config) if hasattr(config, '__dict__') else config.copy()
            saved_config['epochs'] = best_epoch 
            
            best_checkpoint = {
                'model_state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'epoch': best_epoch,
                'config': saved_config,
                'best_val_score': best_val_score,
                'best_val_loss': val_loss,
            }
            print(f"  ✓ New best model found (Score: {best_val_score:.4f})")
        
        if early_stopping(val_score):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_score{best_val_score:.4f}" 
    run_dir = save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    if best_checkpoint:
        best_checkpoint['history'] = history
        torch.save(best_checkpoint, run_dir / 'best_checkpoint.pt')
    
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': vars(config) if hasattr(config, '__dict__') else config,
        'final_val_score': val_score,
        'history': history,
    }
    torch.save(final_checkpoint, run_dir / 'final_checkpoint.pt')
    
    config_dict = {k: str(v) for k, v in type(config).__dict__.items() if not k.startswith('__')}
    config_dict['epochs'] = best_epoch 
    
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"  Best Val Score: {best_val_score:.4f} (Epoch {best_epoch})")
    print(f"  Saved to: {run_dir}")
    print(f"{'='*50}")
    
    return history, run_dir
