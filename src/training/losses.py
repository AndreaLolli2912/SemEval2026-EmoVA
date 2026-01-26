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


def within_user_pearson(pred, targ, user_ids):
    loss = 0
    count = 0

    for u in torch.unique(user_ids):
        idx = user_ids == u
        if idx.sum() < 2:
            continue

        p = pred[idx]
        t = targ[idx]

        p = p - p.mean()
        t = t - t.mean()

        corr = (p * t).sum() / (
            torch.sqrt((p**2).sum() + 1e-8) *
            torch.sqrt((t**2).sum() + 1e-8)
        )
        loss += (1 - corr)
        count += 1

    return loss / max(count, 1)


def between_user_pearson(pred, targ, user_ids):
    preds_u = []
    targs_u = []

    for u in torch.unique(user_ids):
        idx = user_ids == u
        preds_u.append(pred[idx].mean())
        targs_u.append(targ[idx].mean())

    preds_u = torch.stack(preds_u)
    targs_u = torch.stack(targs_u)

    preds_u -= preds_u.mean()
    targs_u -= targs_u.mean()

    corr = (preds_u * targs_u).sum() / (
        torch.sqrt((preds_u**2).sum() + 1e-8) *
        torch.sqrt((targs_u**2).sum() + 1e-8)
    )

    return 1 - corr


def composite_aligned_loss(pred, targ, user_ids):
    L_within = within_user_pearson(pred, targ, user_ids)
    L_between = between_user_pearson(pred, targ, user_ids)

    return 0.6 * L_within + 0.4 * L_between
