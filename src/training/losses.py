def masked_mse_loss(predictions, targets, mask):
    mask = mask.unsqueeze(-1).float()
    squared_error = (predictions - targets) ** 2
    masked_se = squared_error * mask
    n_valid = mask.sum() * 2
    return masked_se.sum() / n_valid.clamp(min=1)

def ccc_loss(predictions, targets, mask):
    """
    Concordance Correlation Coefficient loss.
    
    CCC = (2 * covariance) / (var_pred + var_target + (mean_pred - mean_target)^2)
    
    Loss = 1 - CCC (so we minimize)
    
    This encourages both correlation AND agreement in mean/variance.
    """
    mask = mask.bool()
    
    # Flatten valid predictions
    pred = predictions[mask]  # [N, 2]
    targ = targets[mask]      # [N, 2]
    
    loss = 0
    for dim in range(2):  # valence, arousal
        p = pred[:, dim]
        t = targ[:, dim]
        
        p_mean = p.mean()
        t_mean = t.mean()
        p_var = p.var()
        t_var = t.var()
        covar = ((p - p_mean) * (t - t_mean)).mean()
        
        ccc = (2 * covar) / (p_var + t_var + (p_mean - t_mean) ** 2 + 1e-8)
        loss += (1 - ccc)
    
    return loss / 2  # Average over valence and arousal

def combined_loss(predictions, targets, mask, mse_weight=0.5, ccc_weight=0.5):
    """
    Combine MSE and CCC losses.
    
    - MSE: Penalizes absolute errors (helps with MAE evaluation metric)
    - CCC: Encourages correlation (helps with r evaluation metric)
    """
    mse = masked_mse_loss(predictions, targets, mask)
    ccc = ccc_loss(predictions, targets, mask)
    
    return mse_weight * mse + ccc_weight * ccc