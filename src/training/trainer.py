"""
Training functions for EmoVA.
"""
import json
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from tqdm.auto import tqdm

from src.training.losses import masked_mse_loss, combined_loss
from src.training.utils import EarlyStopping


def train_epoch(
    model,
    dataloader,
    loss_fn,  # "masked_mse_loss" or "combined_loss"
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
    
    # scaler = torch.cuda.amp.GradScaler()  # FP16 safe scaling
    scaler = torch.amp.GradScaler("cuda")
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for step, batch in enumerate(pbar):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)  # added non_blocking=True to speed up the code
        seq_lengths = batch['seq_lengths'].to(device, non_blocking=True)
        seq_mask = batch['seq_attention_mask'].to(device, non_blocking=True)
        valences = batch['valences'].to(device, non_blocking=True)
        arousals = batch['arousals'].to(device, non_blocking=True)
        targets = torch.stack([valences, arousals], dim=-1)
        
        # Forward + loss in AMP
        with torch.amp.autocast('cuda'):
            predictions = model(input_ids, attention_mask, seq_lengths, seq_mask)
            
            if loss_fn == "masked_mse_loss":
                current_loss_raw = masked_mse_loss(predictions.float(), targets.float(), seq_mask)
            elif loss_fn == "combined_loss":
                current_loss_raw = combined_loss(predictions.float(), targets.float(), seq_mask)
            else:
                raise ValueError(f"Unknown loss {loss_fn}")
            
            # scale for gradient accumulation
            current_loss = current_loss_raw / config.accumulation_steps
        
        # Backward with scaler
        scaler.scale(current_loss).backward()

        current_loss_val = current_loss_raw.detach().float().item()  # save the current non scaled loss
        
        # Gradient accumulation step
        if (step + 1) % config.accumulation_steps == 0:
            if clipper:
                scaler.unscale_(optimizer)
                clipper(model)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # Track loss
        with torch.no_grad():
            mask = seq_mask.bool()
            num_valid = mask.sum().item()
            total_loss += (current_loss_val*num_valid)
                        #(current_loss.detach().float().item() * config.accumulation_steps * num_valid)
            total_samples += num_valid
            
            if collect_preds and collected < max_collect_samples:
                preds_cpu = predictions[mask].detach().cpu()
                targs_cpu = targets[mask].detach().cpu()
                remaining = max_collect_samples - collected
                all_preds.append(preds_cpu[:remaining])
                all_targets.append(targs_cpu[:remaining])
                collected += preds_cpu.size(0)
        
        # tqdm display
        #loss_to_show = float(current_loss.detach().float().item() * config.accumulation_steps)
        #pbar.set_postfix({'loss': loss_to_show})
        pbar.set_postfix({'loss': current_loss_val})
        
        # Free memory
        del predictions, targets, input_ids, attention_mask, seq_lengths, seq_mask, current_loss, current_loss_raw, valence, arousal # add from current_loss
        #torch.cuda.empty_cache()
    
    # Handle leftover steps if not divisible by accumulation_steps
    # Gradient accumulation step
    if (step + 1) % config.accumulation_steps == 0:
        scaler.unscale_(optimizer)  # Add this line
        if clipper:
            clipper(model)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    
    torch.cuda.empty_cache()
    
    result = {
        'loss': total_loss / total_samples,
        'grad_norm': (
            np.mean(clipper.grad_norms[-len(dataloader):])
            if clipper and clipper.grad_norms
            else 0.0
        ),
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
    collect_preds=False,
    max_collect_samples=10000,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_preds = [] if collect_preds else None
    all_targets = [] if collect_preds else None
    collected = 0

    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    for batch in pbar:
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        seq_lengths = batch['seq_lengths'].to(device, non_blocking=True)
        seq_mask = batch['seq_attention_mask'].to(device, non_blocking=True)

        valences = batch['valences'].to(device, non_blocking=True)
        arousals = batch['arousals'].to(device, non_blocking=True)
        targets = torch.stack([valences, arousals], dim=-1)

        predictions = model(input_ids, attention_mask, seq_lengths, seq_mask)
        loss = masked_mse_loss(predictions, targets, seq_mask)

        mask = seq_mask.bool()
        num_valid = mask.sum().item()

        total_loss += loss.item() * num_valid
        total_samples += num_valid

        if collect_preds and collected < max_collect_samples:
            preds_cpu = predictions[mask].cpu()
            targs_cpu = targets[mask].cpu()

            remaining = max_collect_samples - collected
            all_preds.append(preds_cpu[:remaining])
            all_targets.append(targs_cpu[:remaining])
            collected += preds_cpu.size(0)

        pbar.set_postfix({'loss': loss.item()})

        del predictions, targets, input_ids, attention_mask, seq_lengths, seq_mask, valence, arousal # added from valcence

    torch.cuda.empty_cache()

    result = {'loss': total_loss / total_samples}

    if collect_preds:
        result['preds'] = torch.cat(all_preds, dim=0)
        result['targets'] = torch.cat(all_targets, dim=0)

    return result

def train(model, train_loader, val_loader, loss_fn, optimizer, scheduler, device, config, clipper=None, save_dir='outputs'):
    """
    Full training loop with checkpoint saving.
    
    Saves to: {save_dir}/{timestamp}_valloss{best_loss}/
        - best_checkpoint.pt
        - final_checkpoint.pt
        - config.json
    
    Args:
        model: The model to train
        train_loader: Training dataloader
        val_loader: Validation dataloader
        loss_fn: Loss function name
        optimizer: Optimizer
        scheduler: LR scheduler (must have .step(val_loss) method)
        device: torch device
        config: Config object with epochs, patience, accumulation_steps
        clipper: Optional GradientClipper
        save_dir: Base directory for saving checkpoints
    
    Returns:
        history: dict with 'train_loss', 'val_loss' lists
        run_dir: Path to the run directory
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_epoch = 0
    best_checkpoint = None
    early_stopping = EarlyStopping(patience=config.patience, mode='min')
    
    for epoch in range(config.epochs):
        
        
        # Train
        train_result = train_epoch(
            model, train_loader, loss_fn, optimizer, device, config, clipper,
            collect_preds=False
        )
        
        # Evaluate
        val_result = eval_epoch(
            model, val_loader, device,
            collect_preds=False
        )
        
        # Update scheduler
        scheduler.step(val_result['loss'])
        
        # Log
        if (epoch + 1) % 5 ==0:
            print(f"\nEpoch {epoch + 1}/{config.epochs}")
            print("-" * 30)
            print(f"  Train Loss: {train_result['loss']:.4f}")
            print(f"  Val Loss:   {val_result['loss']:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Store history
        history['train_loss'].append(train_result['loss'])
        history['val_loss'].append(val_result['loss'])
        
        # Track best model
        if val_result['loss'] < best_val_loss:
            best_val_loss = val_result['loss']
            best_epoch = epoch + 1
            
            # Save state dicts to CPU
            best_checkpoint = {
                'model_state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                'optimizer_state_dict': {k: v if not isinstance(v, torch.Tensor) else v.cpu().clone() 
                                         for k, v in optimizer.state_dict().items()},
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1,
                'config': vars(config) if hasattr(config, '__dict__') else config,
                'best_val_loss': best_val_loss,
            }
            print(f"  ✓ New best model")
        
        # Early stopping check
        if early_stopping(val_result['loss']):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
    
    # Create run directory with timestamp and loss
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_valloss{best_val_loss:.4f}"
    run_dir = save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save best checkpoint with full history
    best_checkpoint['history'] = history
    torch.save(best_checkpoint, run_dir / 'best_checkpoint.pt')
    
    # Save final checkpoint
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch + 1,
        'config': vars(config) if hasattr(config, '__dict__') else config,
        'final_val_loss': val_result['loss'],
        'history': history,
    }
    torch.save(final_checkpoint, run_dir / 'final_checkpoint.pt')
    
    # Save config as JSON (human readable)
    config_dict = vars(config) if hasattr(config, '__dict__') else config
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f} (epoch {best_epoch})")
    print(f"  Final val loss: {val_result['loss']:.4f}")
    print(f"  Saved to: {run_dir}")
    print(f"{'='*50}")
    
    return history, run_dir
