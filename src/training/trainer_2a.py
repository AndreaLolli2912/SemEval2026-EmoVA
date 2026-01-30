"""
Training functions for EmoVA.
"""
import json
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from tqdm.auto import tqdm

# Import your new losses
from src.training.losses import masked_mse_loss, combined_loss, ccc_loss
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

    criterion = nn.MSELoss()

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(dataloader, desc="Training Task 2a", leave=False)
    
    for step, batch in enumerate(pbar):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        history_va = batch['history'].to(device, non_blocking=True)
        seq_lengths = batch['seq_lengths'].to(device, non_blocking=True)
        
        targets = batch['targets'].to(device, non_blocking=True)
        
        # --- 1. EXTRACT WEIGHTS FROM CONFIG ---
        
        with torch.amp.autocast('cuda'):
            predictions = model(input_ids, attention_mask, history_va, seq_lengths)
            
            # --- 2. PASS WEIGHTS TO LOSS FUNCTIONS ---
           current_loss_raw = criterion(predictions, targets)
             
            
            current_loss = current_loss_raw / config.accumulation_steps
        
        scaler.scale(current_loss).backward()
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
        del predictions, targets, input_ids, attention_mask, seq_lengths, history_va, current_loss
    
    # Handle remaining steps
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
        'grad_norm': np.mean(clipper.grad_norms[-len(dataloader):]) if clipper and clipper.grad_norms else 0.0,
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
    config, # Added config to access shares if needed for loss calculation
    collect_preds=False, # Kept signature for compatibility, but logic is changed below
):
    """
    Evaluates model using official SemEval metrics.
    Accumulates all predictions to compute user-level correlations.
    """
    model.eval()

    total_loss = 0.0
    total_sample = 0.0
    criterion = nn.MSELoss()

    all_preds = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc="Evaluating Task 2a", leave=False)
    
    for batch in pbar:
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        seq_lengths = batch['seq_lengths'].to(device, non_blocking=True)
        history_va = batch['history'].to(device, non_blocking=True)
        user_ids = batch['user_ids'] # List of strings

        targets = batch['targets'].to(device,  non_blocking=True)

        predictions = model(input_ids, attention_mask, history_va, seq_lengths)
        
        # 1. Compute Loss (for Scheduler)
        loss = criterion(predictions, targets)

        
        batch_size = input_ids.size(0)
        total_loss +=loss
        total_samples +=batch_size
        
        # Collect prediction for metrics
        pred_cpu = preds_cpu = predictions.cpu().numpy()   # [Batch, 2]
        targets_cpu = targets.cpu().numpy()     # [Batch, 2]
        
        all_preds.append(preds_cpu)
        all_targets.append(targets_cpu)

        pbar.set_postfix({'loss': loss.item()})
        
        # Cleanup
        del predictions, targets, input_ids, attention_mask, seq_lengths, history_va

    torch.cuda.empty_cache()

    # --- Finalize SemEval Metrics ---
    # Concatenate lists into numpy arrays for each user
    final_preds = np.concatenate(all_preds, axis=0)
    final_gold = np.concatenate(all_targets, axis=0)
    
    # Compute Metrics (MSE per component)
    mse_valence = np.mean((final_preds[:, 0] - final_gold[:, 0])**2)
    mse_arousal = np.mean((final_preds[:, 1] - final_gold[:, 1])**2)
    overall_mse = (mse_valence + mse_arousal) / 2.0

    def pearson_corr(x, y):
        if np.std(x) == 0 or np.std(y) == 0: return 0.0
        return np.corrcoef(x, y)[0, 1]

    corr_valence = pearson_corr(final_preds[:, 0], final_gold[:, 0])
    corr_arousal = pearson_corr(final_preds[:, 1], final_gold[:, 1])
    overall_corr = (corr_valence + corr_arousal) / 2.0
    result = {
        'valence/mse': mse_valence,
        'arousal/mse': mse_arousal,
        'overall/mse': overall_mse,
        'valence/corr': corr_valence,
        'arousal/corr': corr_arousal,
        'overall/r_composite': overall_corr # Using Correlation as "r_composite" equivalent
    }

    return result

def train(model, train_loader, val_loader, loss_fn_name, optimizer, scheduler, device, config, clipper=None, save_dir='outputs_task2a'):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    n_epochs = config.epochs
    history = {'train_loss': [], 'val_loss': [], 'val_score': []}
    
    # Track BEST SCORE (Higher is better), not best loss
    best_val_score = -1.0 
    best_epoch = 0
    best_checkpoint = None
    
    # Early stopping usually monitors loss, but for this task, 
    # monitoring the composite score is safer.
    # We invert score because EarlyStopping expects 'min' by default, 
    # or we can set mode='max'. Let's assume your EarlyStopping supports mode='max'.
    early_stopping = EarlyStopping(patience=config.patience, mode='max')
    
    for epoch in range(config.epochs):
        
        # Train
        train_result = train_epoch(
            model, train_loader, loss_fn_name, optimizer, device, config, clipper
        )
        
        # Evaluate
        val_result = eval_epoch(
            model, val_loader, device, config
        )

        train_loss = train_result['loss']
        val_loss = val_result['loss']
        val_score = val_result['score']
        metrics = val_result['metrics']
        
        # Update scheduler (Scheduler usually likes minimizing Loss)
        # If your scheduler is ReduceLROnPlateau, keep passing loss.
        if scheduler:
            scheduler.step(val_loss)
        
        # Log
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        print("-" * 30)
        print(f"  Train Loss:   {train_result['loss']:.4f}")
        print(f"  Val Loss:     {val_loss:.4f}")
        print(f"  Val Score:    {val_score:.4f} (Composite r)")
        print(f"  > Valence r:  {metrics['valence/r_composite']:.4f}")
        print(f"  > Arousal r:  {metrics['arousal/r_composite']:.4f}")
        print(f"  LR: {optimizer.param_groups[-1]['lr']:.2e}")
        
        # Store history
        history['train_loss'].append(train_result['loss'])
        history['val_loss'].append(val_loss)
        history['val_score'].append(val_score)
        
        # Track best model based on SCORE
        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch + 1
            
            saved_config = vars(config) if hasattr(config, '__dict__') else config.copy()
            saved_config['epochs'] = best_epoch  # overwrite with stopping epoch 
            
            best_checkpoint = {
                'model_state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': best_epoch,
                'config': saved_config,  # updated config
                'best_val_score': best_val_score,
                'best_val_loss': val_loss,
            }
            print(f"  ✓ New best model found (Score: {best_val_score:.4f})")
        
        # Early stopping check (monitor Score)
        if early_stopping(val_score):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
    
    # Save best model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Name folder by SCORE now
    run_name = f"{timestamp}_score{best_val_score:.4f}" 
    run_dir = save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    best_checkpoint['history'] = history
    torch.save(best_checkpoint, run_dir / 'best_checkpoint.pt')
    
    # Save final checkpoint
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': vars(config) if hasattr(config, '__dict__') else config,
        'final_val_score': val_score,
        'history': history,
    }
    torch.save(final_checkpoint, run_dir / 'final_checkpoint.pt')
    
    # Save config
    # --- SAVE CONFIG (SIMPLE FIX) ---
    # 1. Grab attributes from the class type, filter out internal python stuff
    config_dict = {k: str(v) for k, v in type(config).__dict__.items() if not k.startswith('__')}
    
    # 2. Update the epoch count
    config_dict['epochs'] = best_epoch 
    
    # 3. Save
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"  Best Val Score: {best_val_score:.4f} (Epoch {best_epoch})")
    print(f"  Saved to: {run_dir}")
    print(f"{'='*50}")
    
    return history, run_dir
