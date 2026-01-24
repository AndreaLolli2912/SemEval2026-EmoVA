"""
Training functions for EmoVA.
"""
import json
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from tqdm.auto import tqdm

from src.training.losses import masked_mse_loss
from src.training.utils import EarlyStopping


def train_epoch(model, dataloader, optimizer, device, config, clipper=None):
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        device: torch device
        config: Config object with accumulation_steps
        clipper: Optional GradientClipper
    
    Returns:
        dict with 'loss', 'grad_norm', 'preds', 'targets'
    """
    model.train()
    
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        seq_lengths = batch['seq_lengths'].to(device)
        seq_mask = batch['seq_attention_mask'].to(device)
        
        valences = batch['valences'].to(device)
        arousals = batch['arousals'].to(device)
        targets = torch.stack([valences, arousals], dim=-1)
        
        predictions = model(input_ids, attention_mask, seq_lengths, seq_mask)
        
        loss = masked_mse_loss(predictions, targets, seq_mask)
        loss = loss / config.accumulation_steps
        loss.backward()
        
        if (step + 1) % config.accumulation_steps == 0:
            if clipper:
                clipper(model)
            optimizer.step()
            optimizer.zero_grad()
        
        with torch.no_grad():
            mask = seq_mask.bool()
            num_valid = mask.sum().item()
            total_loss += loss.item() * config.accumulation_steps * num_valid
            total_samples += num_valid
            
            all_preds.append(predictions[mask].cpu())
            all_targets.append(targets[mask].cpu())
        
        pbar.set_postfix({'loss': loss.item() * config.accumulation_steps})
    
    # Handle leftover gradients
    if (step + 1) % config.accumulation_steps != 0:
        if clipper:
            clipper(model)
        optimizer.step()
        optimizer.zero_grad()
    
    return {
        'loss': total_loss / total_samples,
        'grad_norm': np.mean(clipper.grad_norms[-len(dataloader):]) if clipper and clipper.grad_norms else 0.0,
        'preds': torch.cat(all_preds, dim=0),
        'targets': torch.cat(all_targets, dim=0),
    }


@torch.no_grad()
def eval_epoch(model, dataloader, device):
    """
    Evaluate for one epoch.
    
    Args:
        model: The model to evaluate
        dataloader: Validation dataloader
        device: torch device
    
    Returns:
        dict with 'loss', 'preds', 'targets'
    """
    model.eval()
    
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        seq_lengths = batch['seq_lengths'].to(device)
        seq_mask = batch['seq_attention_mask'].to(device)
        
        valences = batch['valences'].to(device)
        arousals = batch['arousals'].to(device)
        targets = torch.stack([valences, arousals], dim=-1)
        
        predictions = model(input_ids, attention_mask, seq_lengths, seq_mask)
        loss = masked_mse_loss(predictions, targets, seq_mask)
        
        mask = seq_mask.bool()
        num_valid = mask.sum().item()
        total_loss += loss.item() * num_valid
        total_samples += num_valid
        
        all_preds.append(predictions[mask].cpu())
        all_targets.append(targets[mask].cpu())
        
        pbar.set_postfix({'loss': loss.item()})
    
    return {
        'loss': total_loss / total_samples,
        'preds': torch.cat(all_preds, dim=0),
        'targets': torch.cat(all_targets, dim=0),
    }


def train(model, train_loader, val_loader, optimizer, scheduler, device, config, clipper=None, save_dir='outputs'):
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
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        print("-" * 30)
        
        # Train
        train_result = train_epoch(model, train_loader, optimizer, device, config, clipper)
        
        # Evaluate
        val_result = eval_epoch(model, val_loader, device)
        
        # Update scheduler
        scheduler.step(val_result['loss'])
        
        # Log
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
