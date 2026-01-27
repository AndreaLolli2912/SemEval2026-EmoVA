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

# Import the evaluation logic
from src.evaluation.metrics import evaluate_subtask1

def train_epoch(
    model,
    dataloader,
    loss_fn_name,  # "masked_mse_loss", "combined_loss", or "ccc_loss"
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
    
    all_preds = [] if collect_preds else None
    all_targets = [] if collect_preds else None
    collected = 0
    
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler("cuda")
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for step, batch in enumerate(pbar):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        seq_lengths = batch['seq_lengths'].to(device, non_blocking=True)
        seq_mask = batch['seq_attention_mask'].to(device, non_blocking=True)
        valences = batch['valences'].to(device, non_blocking=True)
        arousals = batch['arousals'].to(device, non_blocking=True)
        
        targets = torch.stack([valences, arousals], dim=-1)
        
        # --- 1. EXTRACT WEIGHTS FROM CONFIG ---
        # Defaults: 20% MSE / 80% CCC; 50% Valence / 50% Arousal
        mse_share = getattr(config, 'mse_share', 0.2)
        valence_share = getattr(config, 'valence_share', 0.5)

        with torch.amp.autocast('cuda'):
            predictions = model(input_ids, attention_mask, seq_lengths, seq_mask)
            
            # --- 2. PASS WEIGHTS TO LOSS FUNCTIONS ---
            if loss_fn_name == "masked_mse_loss":
                current_loss_raw = masked_mse_loss(
                    predictions.float(), 
                    targets.float(), 
                    seq_mask, 
                    valence_share=valence_share
                )
            elif loss_fn_name == "ccc_loss":
                current_loss_raw = ccc_loss(
                    predictions.float(), 
                    targets.float(), 
                    seq_mask, 
                    valence_share=valence_share
                )
            elif loss_fn_name == "combined_loss":
                current_loss_raw = combined_loss(
                    predictions.float(), 
                    targets.float(), 
                    seq_mask, 
                    mse_share=mse_share,       
                    valence_share=valence_share
                )
            else:
                raise ValueError(f"Unknown loss {loss_fn_name}")
            
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
            mask = seq_mask.bool()
            num_valid = mask.sum().item()
            total_loss += (current_loss_val * num_valid)
            total_samples += num_valid
            
            if collect_preds and collected < max_collect_samples:
                preds_cpu = predictions[mask].detach().cpu()
                targs_cpu = targets[mask].detach().cpu()
                remaining = max_collect_samples - collected
                all_preds.append(preds_cpu[:remaining])
                all_targets.append(targs_cpu[:remaining])
                collected += preds_cpu.size(0)
        
        pbar.set_postfix({'loss': current_loss_val})
        del predictions, targets, input_ids, attention_mask, seq_lengths, seq_mask, current_loss
    
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
    total_samples = 0

    # We MUST collect all predictions to compute the SemEval metric
    # Dictionary to store per-user history: user_id -> list of predictions
    user_preds_accumulator = {}
    user_gold_accumulator = {}
    
    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    
    for batch in pbar:
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        seq_lengths = batch['seq_lengths'].to(device, non_blocking=True)
        seq_mask = batch['seq_attention_mask'].to(device, non_blocking=True)
        valences = batch['valences'].to(device, non_blocking=True)
        arousals = batch['arousals'].to(device, non_blocking=True)
        user_ids = batch['user_ids'] # List of strings

        targets = torch.stack([valences, arousals], dim=-1)

        predictions = model(input_ids, attention_mask, seq_lengths, seq_mask)
        
        # 1. Compute Loss (for Scheduler)
        # Use simple MSE for logging, or consistent loss if preferred
        # Here we use masked_mse_loss for interpretability (absolute error)
        loss = masked_mse_loss(predictions, targets, seq_mask, valence_share=0.5)

        mask = seq_mask.bool()
        num_valid = mask.sum().item()

        total_loss += loss.item() * num_valid
        total_samples += num_valid

        # 2. Collect Predictions for SemEval Metric
        # We need to unroll the batch into user histories
        preds_cpu = predictions.cpu().numpy()
        targets_cpu = targets.cpu().numpy()
        seq_lengths_cpu = seq_lengths.cpu().numpy()
        
        batch_size = preds_cpu.shape[0]
        
        for i in range(batch_size):
            uid = user_ids[i]
            slen = seq_lengths_cpu[i]
            
            # Slice valid texts for this user in this batch
            u_pred = preds_cpu[i, :slen, :] # (Seq_Len, 2)
            u_gold = targets_cpu[i, :slen, :] # (Seq_Len, 2)
            
            if uid not in user_preds_accumulator:
                user_preds_accumulator[uid] = []
                user_gold_accumulator[uid] = []
            
            user_preds_accumulator[uid].append(u_pred)
            user_gold_accumulator[uid].append(u_gold)

        pbar.set_postfix({'loss': loss.item()})
        
        # Cleanup
        del predictions, targets, input_ids, attention_mask, seq_lengths, seq_mask, valences, arousals

    torch.cuda.empty_cache()

    # --- Finalize SemEval Metrics ---
    # Concatenate lists into numpy arrays for each user
    final_preds = {k: np.concatenate(v, axis=0) for k, v in user_preds_accumulator.items()}
    final_gold = {k: np.concatenate(v, axis=0) for k, v in user_gold_accumulator.items()}
    
    # Run Official Evaluation
    metrics = evaluate_subtask1(final_preds, final_gold, verbose=False)

    result = {
        'loss': total_loss / max(total_samples, 1),
        'metrics': metrics,
        'score': metrics['overall/r_composite'] # The Ranking Score
    }

    return result

def train(model, train_loader, val_loader, loss_fn_name, optimizer, scheduler, device, config, clipper=None, save_dir='outputs'):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
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
        
        val_loss = val_result['loss']
        val_score = val_result['score']
        metrics = val_result['metrics']
        
        # Update scheduler (Scheduler usually likes minimizing Loss)
        # If your scheduler is ReduceLROnPlateau, keep passing loss.
        if scheduler:
            scheduler.step(val_loss)
        
        # Log
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
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
    config_dict = vars(config) if hasattr(config, '__dict__') else config
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"  Best Val Score: {best_val_score:.4f} (Epoch {best_epoch})")
    print(f"  Saved to: {run_dir}")
    print(f"{'='*50}")
    
    return history, run_dir