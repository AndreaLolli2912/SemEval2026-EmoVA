import torch

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
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
    """Gradient clipping utility with monitoring."""
    
    def __init__(self, max_norm=1.0):
        self.max_norm = max_norm
        self.grad_norms = []
        
    def __call__(self, model):
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
        self.grad_norms.append(norm.item())
        return norm