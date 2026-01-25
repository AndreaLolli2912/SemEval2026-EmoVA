"""
Training utilities for EmoVA.
"""
import torch


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        mode: 'min' or 'max'
    """
    def __init__(self, patience=10, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class GradientClipper:
    """
    Gradient clipping with tracking.
    
    Args:
        max_norm: Maximum gradient norm
    """
    def __init__(self, max_norm=1.0):
        self.max_norm = max_norm
        self.grad_norms = []
    
    def __call__(self, model):
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
        self.grad_norms.append(norm.item())
        return norm


def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """
    Load model with correct config from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        model: Loaded model in eval mode
        config: Config dict used for training
        checkpoint: Full checkpoint dict
    
    Usage:
        model, config, checkpoint = load_model_from_checkpoint('outputs/run/best_checkpoint.pt')
        print(f"Loaded from epoch {checkpoint['epoch']}, val_loss={checkpoint['best_val_loss']:.4f}")
    """
    from src.models import AffectModel
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Rebuild model with saved config
    model = AffectModel(
        model_path=config['model_name'],
        isab_inducing_points=config['isab_inducing_points'],
        pma_num_seeds=config['pma_num_seeds'],
        lstm_hidden_dim=config['lstm_hidden_dim'],
        lstm_num_layers=config['lstm_num_layers'],
        dropout=config['dropout'],
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"  Val loss: {checkpoint.get('best_val_loss', checkpoint.get('final_val_loss', 'N/A')):.4f}")
    
    return model, config, checkpoint


def resume_training(checkpoint_path, model, optimizer, scheduler, device='cuda'):
    """
    Resume training from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance (must match architecture)
        optimizer: Optimizer instance
        scheduler: Scheduler instance
        device: Device
    
    Returns:
        start_epoch: Epoch to resume from
        history: Training history so far
    
    Usage:
        start_epoch, history = resume_training('checkpoint.pt', model, optimizer, scheduler)
        # Then continue training loop from start_epoch
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch']
    history = checkpoint.get('history', {'train_loss': [], 'val_loss': []})
    
    print(f"Resumed from epoch {start_epoch}")
    
    return start_epoch, history