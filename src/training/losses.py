import torch

def masked_mse_loss(predictions, targets, mask, valence_share=0.5):
    """
    Computes weighted MSE on Valid Documents.
    
    Args:
        predictions: (Batch, Num_Texts, 2)
        targets:     (Batch, Num_Texts, 2)
        mask:        (Batch, Num_Texts) -> Binary 1/0 for Real/Padding
        valence_share: (float) 0.0 to 1.0
    """
    # Define weights summing to 1.0
    arousal_share = 1.0 - valence_share
    dim_weights = [valence_share, arousal_share]
        
    mask = mask.float()
    
    # Sum of valid documents in the batch (not tokens)
    n_valid_texts = mask.sum().clamp(min=1) 
    
    loss = 0.0
    
    for i in range(2): # 0=Valence, 1=Arousal
        pred_dim = predictions[:, :, i]
        targ_dim = targets[:, :, i]
        
        # Calculate Squared Error
        squared_error = (pred_dim - targ_dim) ** 2
        
        # Zero out padding errors
        # mask is (Batch, Num_Texts), matches pred_dim dimensions
        masked_se = squared_error * mask
        
        # Mean MSE per valid document
        dim_loss = masked_se.sum() / n_valid_texts
        
        # Add weighted contribution
        loss += dim_loss * dim_weights[i]
        
    return loss 

def ccc_loss(predictions, targets, mask, valence_share=0.5):
    """
    Computes weighted CCC loss on Valid Documents.
    """
    arousal_share = 1.0 - valence_share
    dim_weights = [valence_share, arousal_share]

    mask = mask.bool()
    
    # Flatten: Select only valid texts from the batch
    # Result Shape: (Total_Valid_Texts_In_Batch, 2)
    pred = predictions[mask]
    targ = targets[mask]
    
    loss = 0.0
    
    for i in range(2): 
        p = pred[:, i]
        t = targ[:, i]
        
        # Statistics over the valid population
        p_mean = p.mean()
        t_mean = t.mean()
        p_var = p.var(unbiased=False)
        t_var = t.var(unbiased=False)
        covar = ((p - p_mean) * (t - t_mean)).mean()
        
        # CCC Calculation
        numerator = 2 * covar
        denominator = p_var + t_var + (p_mean - t_mean) ** 2 + 1e-8
        ccc = numerator / denominator
        
        # Add weighted contribution
        loss += (1.0 - ccc) * dim_weights[i]
    
    return loss

def combined_loss(predictions, targets, mask, mse_share=0.2, valence_share=0.5):
    """
    Total Loss = (MSE * mse_share) + (CCC * (1 - mse_share))
    """
    ccc_share = 1.0 - mse_share
    
    mse = masked_mse_loss(predictions, targets, mask, valence_share=valence_share)
    ccc = ccc_loss(predictions, targets, mask, valence_share=valence_share)
    
    return (mse * mse_share) + (ccc * ccc_share)