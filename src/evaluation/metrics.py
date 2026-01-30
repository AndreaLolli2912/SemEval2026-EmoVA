"""
SemEval 2026 EmoVA - Official Evaluation Metrics
=================================================

Subtask 1: Longitudinal Affect Assessment

Metrics:
- Between-user correlation: Pearson r on per-user means
- Within-user correlation: Mean of per-user Pearson r
- Composite correlation: Fisher z-transform combination (ranking metric)

Usage:
    from src.evaluation.metrics import evaluate_subtask1, collect_predictions_for_eval
    
    predictions, gold = collect_predictions_for_eval(model, dataloader, device)
    results = evaluate_subtask1(predictions, gold)
    print_evaluation_results(results)
"""

import torch
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
import logging

# Setup logging
logger = logging.getLogger(__name__)


# =============================================================================
# Core Statistical Functions
# =============================================================================

def pearson_correlation(pred: np.ndarray, gold: np.ndarray) -> float:
    """
    Compute Pearson correlation coefficient.
    
    Args:
        pred: 1D array of predictions
        gold: 1D array of gold values
    
    Returns:
        Pearson r, or np.nan if insufficient data
    """
    if len(pred) < 2:
        logger.warning(f"Insufficient data for correlation: n={len(pred)}")
        return np.nan
    
    # Check for zero variance
    if np.std(pred) < 1e-10 or np.std(gold) < 1e-10:
        logger.warning("Zero variance detected, correlation undefined")
        return np.nan
    
    r, p_value = stats.pearsonr(pred, gold)
    return r


def mae(pred: np.ndarray, gold: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(pred - gold))


def fisher_z_mean(r1: float, r2: float) -> float:
    """
    Combine two correlations using Fisher's z-transformation.
    
    r_composite = tanh((arctanh(r1) + arctanh(r2)) / 2)
    
    Args:
        r1: First correlation coefficient
        r2: Second correlation coefficient
    
    Returns:
        Combined correlation using Fisher z-transform
    """
    # Handle NaN inputs
    if np.isnan(r1) or np.isnan(r2):
        logger.warning(f"NaN input to fisher_z_mean: r1={r1}, r2={r2}")
        return np.nan
    
    # Clip to avoid infinity at ±1
    r1 = np.clip(r1, -0.9999, 0.9999)
    r2 = np.clip(r2, -0.9999, 0.9999)
    
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    z_mean = (z1 + z2) / 2
    
    return np.tanh(z_mean)


# =============================================================================
# Between-User Metrics
# =============================================================================

def between_user_correlation(
    predictions: Dict[str, np.ndarray],
    gold: Dict[str, np.ndarray],
    verbose: bool = False
) -> float:
    """
    Compute between-user Pearson correlation.
    
    For each user, compute mean predicted and mean gold scores,
    then correlate these means across users.
    
    Args:
        predictions: user_id -> array of predictions for that user
        gold: user_id -> array of gold values for that user
        verbose: If True, log debug information
    
    Returns:
        Pearson correlation of user means
    """
    user_ids = list(predictions.keys())
    n_users = len(user_ids)
    
    pred_means = np.array([predictions[u].mean() for u in user_ids])
    gold_means = np.array([gold[u].mean() for u in user_ids])
    
    if verbose:
        logger.info(f"Between-user correlation: {n_users} users")
        logger.info(f"  Pred means range: [{pred_means.min():.3f}, {pred_means.max():.3f}]")
        logger.info(f"  Gold means range: [{gold_means.min():.3f}, {gold_means.max():.3f}]")
    
    return pearson_correlation(pred_means, gold_means)


def between_user_mae(
    predictions: Dict[str, np.ndarray],
    gold: Dict[str, np.ndarray]
) -> float:
    """
    Compute between-user MAE.
    
    MAE between per-user mean predictions and per-user mean gold values.
    """
    user_ids = list(predictions.keys())
    
    pred_means = np.array([predictions[u].mean() for u in user_ids])
    gold_means = np.array([gold[u].mean() for u in user_ids])
    
    return mae(pred_means, gold_means)


# =============================================================================
# Within-User Metrics
# =============================================================================

def within_user_correlation(
    predictions: Dict[str, np.ndarray],
    gold: Dict[str, np.ndarray],
    min_texts: int = 2,
    verbose: bool = False
) -> float:
    """
    Compute within-user Pearson correlation.
    
    For each user, compute correlation between predicted and gold text scores,
    then average these correlations across users.
    
    Args:
        predictions: user_id -> array of predictions for that user's texts
        gold: user_id -> array of gold values for that user's texts
        min_texts: minimum texts required per user to compute correlation
        verbose: If True, log debug information
    
    Returns:
        Mean of per-user correlations
    """
    correlations = []
    skipped_users = 0
    nan_correlations = 0
    
    for user_id in predictions.keys():
        pred = predictions[user_id]
        g = gold[user_id]
        
        if len(pred) < min_texts:
            skipped_users += 1
            continue
        
        r = pearson_correlation(pred, g)
        
        if np.isnan(r):
            nan_correlations += 1
            continue
        
        correlations.append(r)
    
    if verbose:
        logger.info(f"Within-user correlation:")
        logger.info(f"  Valid users: {len(correlations)}")
        logger.info(f"  Skipped (< {min_texts} texts): {skipped_users}")
        logger.info(f"  NaN correlations: {nan_correlations}")
        if correlations:
            logger.info(f"  Correlation range: [{min(correlations):.3f}, {max(correlations):.3f}]")
    
    if len(correlations) == 0:
        logger.warning("No valid within-user correlations computed")
        return np.nan
    
    return np.mean(correlations)


def within_user_mae(
    predictions: Dict[str, np.ndarray],
    gold: Dict[str, np.ndarray]
) -> float:
    """
    Compute within-user MAE.
    
    For each user, compute MAE between predicted and gold text scores,
    then average across users.
    """
    maes = []
    
    for user_id in predictions.keys():
        pred = predictions[user_id]
        g = gold[user_id]
        user_mae = mae(pred, g)
        maes.append(user_mae)
    
    return np.mean(maes)


# =============================================================================
# Composite Metrics (used for ranking)
# =============================================================================

def composite_correlation(r_within: float, r_between: float) -> float:
    """
    Compute composite correlation using Fisher's z-transformation.
    
    r_composite = tanh((arctanh(r_within) + arctanh(r_between)) / 2)
    
    This is the official ranking metric for SemEval 2026 EmoVA Subtask 1.
    """
    return fisher_z_mean(r_within, r_between)


def composite_mae(mae_within: float, mae_between: float) -> float:
    """
    Compute composite MAE using Fisher's z-transformation.
    
    Note: This assumes MAE values are in a range suitable for arctanh.
    Values are clipped to [0, 0.9999] to avoid numerical issues.
    """
    mae_within = np.clip(mae_within, 0, 0.9999)
    mae_between = np.clip(mae_between, 0, 0.9999)
    
    return fisher_z_mean(mae_within, mae_between)


# =============================================================================
# Main Evaluation Function
# =============================================================================

def evaluate_subtask1(
    predictions: Dict[str, np.ndarray],
    gold: Dict[str, np.ndarray],
    min_texts_for_within: int = 2,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Complete evaluation for Subtask 1: Longitudinal Affect Assessment.
    
    Args:
        predictions: Dict mapping user_id to array of shape [n_texts, 2]
                    where columns are [valence, arousal]
        gold: Dict mapping user_id to array of shape [n_texts, 2]
        min_texts_for_within: Minimum texts needed to compute within-user correlation
        verbose: If True, log detailed debug information
    
    Returns:
        Dictionary with all metrics for both valence and arousal:
        - {dim}/r_between: Between-user Pearson correlation
        - {dim}/r_within: Within-user Pearson correlation (mean)
        - {dim}/r_composite: Composite correlation (ranking metric)
        - {dim}/mae_between: Between-user MAE
        - {dim}/mae_within: Within-user MAE
        - overall/r_composite: Mean of valence and arousal composite
    """
    # Validation
    assert len(predictions) == len(gold), \
        f"Mismatch in number of users: {len(predictions)} vs {len(gold)}"
    assert set(predictions.keys()) == set(gold.keys()), \
        "User IDs don't match between predictions and gold"
    
    if verbose:
        n_users = len(predictions)
        total_texts = sum(p.shape[0] for p in predictions.values())
        texts_per_user = [p.shape[0] for p in predictions.values()]
        logger.info(f"Evaluating {n_users} users, {total_texts} total texts")
        logger.info(f"Texts per user: min={min(texts_per_user)}, max={max(texts_per_user)}, "
                   f"mean={np.mean(texts_per_user):.1f}")
    
    results = {}
    
    for dim, dim_name in enumerate(['valence', 'arousal']):
        if verbose:
            logger.info(f"\n{'='*40}")
            logger.info(f"Evaluating {dim_name.upper()}")
            logger.info(f"{'='*40}")
        
        # Extract dimension-specific predictions
        pred_dim = {u: p[:, dim] for u, p in predictions.items()}
        gold_dim = {u: g[:, dim] for u, g in gold.items()}
        
        # Validation: check for valid ranges
        if verbose:
            all_pred = np.concatenate(list(pred_dim.values()))
            all_gold = np.concatenate(list(gold_dim.values()))
            logger.info(f"Prediction range: [{all_pred.min():.3f}, {all_pred.max():.3f}]")
            logger.info(f"Gold range: [{all_gold.min():.3f}, {all_gold.max():.3f}]")
        
        # Between-user metrics
        r_between = between_user_correlation(pred_dim, gold_dim, verbose=verbose)
        mae_between = between_user_mae(pred_dim, gold_dim)
        
        # Within-user metrics
        r_within = within_user_correlation(pred_dim, gold_dim, min_texts_for_within, verbose=verbose)
        mae_within = within_user_mae(pred_dim, gold_dim)
        
        # Composite metrics (used for ranking)
        r_composite = composite_correlation(r_within, r_between)
        
        # Store results
        results[f'{dim_name}/r_between'] = r_between
        results[f'{dim_name}/r_within'] = r_within
        results[f'{dim_name}/r_composite'] = r_composite
        results[f'{dim_name}/mae_between'] = mae_between
        results[f'{dim_name}/mae_within'] = mae_within
    
    # Overall composite (average of valence and arousal composites)
    results['overall/r_composite'] = (
        results['valence/r_composite'] + results['arousal/r_composite']
    ) / 2
    
    return results

def evaluate_subtask2a(
    predictions: Dict[str, np.ndarray],
    gold: Dict[str, np.ndarray],
    min_samples: int = 2,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Complete evaluation for Subtask 2a (Forecasting)
    
    Args:
        predictions: Dict mapping user_id to array of shape [n_texts, 2] (predicted state chage)
        gold: Dict mapping user_id to array of shape [n_windows, 2] (Actual State Change)
        verbose: If True, log detailed debug information
    """
    # Validation
    
    assert set(predictions.keys()) == set(gold.keys()), \
        "User IDs don't match between predictions and gold"
    
    if verbose:
        n_users = len(predictions)
        total_texts = sum(p.shape[0] for p in predictions.values())
        texts_per_user = [p.shape[0] for p in predictions.values()]
        logger.info(f"Evaluating {n_users} users, {total_texts} total texts")
        logger.info(f"Texts per user: min={min(texts_per_user)}, max={max(texts_per_user)}, "
                   f"mean={np.mean(texts_per_user):.1f}")
    
    results = {}
    metrics_storage = {'valence': {'r': [], 'mae': []},
                      'arousal': {'r': [], 'mae': []}
                     }
    skipped_users = 0
    for user_id in predictions.keys():
        u_pred = predictions[user_id]
        u_gold = gold[user_id]
        
        # Check samples count
        if len(u_pred) < min_samples:
            skipped_users += 1
            continue
    
        for dim, dim_name in enumerate(['valence', 'arousal']):
            if verbose:
                logger.info(f"\n{'='*40}")
                logger.info(f"Evaluating {dim_name.upper()}")
                logger.info(f"{'='*40}")
                
            # Pearson r
            r = pearson_correlation(u_pred[:, dim], u_gold[:, dim])
            if not np.isnan(r):
                metrics_storage[dim_name]['r'].append(r)
            m = mae(u_pred[:, dim], u_gold[:, dim])
            metrics_storage[dim_name]['mae'].append(m)

        
    for dim in ['valence', 'arousal']:
        # Average Pearson r
        avg_r = np.mean(metrics_storage[dim]['r']) if metrics_storage[dim]['r'] else 0.0
        # Average MAE
        avg_mae = np.mean(metrics_storage[dim]['mae']) if metrics_storage[dim]['mae'] else 0.0
        
        results[f'{dim}/r_per_user'] = avg_r
        results[f'{dim}/mae'] = avg_mae

    # Overall Score (Average of Valence R and Arousal R)
    results['overall/score'] = (results['valence/r_per_user'] + results['arousal/r_per_user']) / 2.0
    
    return results

# =============================================================================
# Helper: Convert batch predictions to evaluation format
# =============================================================================

def collect_predictions_for_eval(
    model,
    dataloader,
    device,
    verbose: bool = False
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Run model on dataloader and collect predictions in evaluation format.
    
    Properly handles padding using seq_lengths from the batch.
    
    Args:
        model: The trained model
        dataloader: DataLoader yielding batches
        device: torch device
        verbose: If True, log debug information
    
    Returns:
        predictions: Dict[user_id -> np.ndarray of shape [n_texts, 2]]
        gold: Dict[user_id -> np.ndarray of shape [n_texts, 2]]
    """
    model.eval()
    
    all_predictions = {}
    all_gold = {}
    
    total_texts = 0
    total_padded = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            seq_lengths = batch['seq_lengths'].to(device)
            seq_mask = batch['seq_attention_mask'].to(device)
            
            batch_size = input_ids.size(0)
            max_seq_len = input_ids.size(1)
            
            # Forward pass
            preds = model(input_ids, attention_mask, seq_lengths, seq_mask)
            preds = preds.cpu().numpy()  # [B, S, 2]
            
            # Gold values
            valences = batch['valences'].numpy()  # [B, S]
            arousals = batch['arousals'].numpy()  # [B, S]
            
            # Debug: track padding
            valid_positions = seq_mask.sum().item()
            total_positions = seq_mask.numel()
            total_texts += valid_positions
            total_padded += (total_positions - valid_positions)
            
            # Collect per user, respecting seq_lengths
            for i, user_id in enumerate(batch['user_ids']):
                seq_len = batch['seq_lengths'][i].item()
                
                # Slice only valid (non-padded) positions
                user_preds = preds[i, :seq_len, :]  # [seq_len, 2]
                user_valence = valences[i, :seq_len]  # [seq_len]
                user_arousal = arousals[i, :seq_len]  # [seq_len]
                user_gold = np.stack([user_valence, user_arousal], axis=-1)  # [seq_len, 2]
                
                # Validation
                assert user_preds.shape == user_gold.shape, \
                    f"Shape mismatch for user {user_id}: pred={user_preds.shape}, gold={user_gold.shape}"
                assert user_preds.shape[0] == seq_len, \
                    f"Seq length mismatch for user {user_id}: expected {seq_len}, got {user_preds.shape[0]}"
                
                # Store (handle case where same user might appear multiple times)
                if user_id in all_predictions:
                    logger.warning(f"User {user_id} appears multiple times in dataloader")
                    all_predictions[user_id] = np.concatenate(
                        [all_predictions[user_id], user_preds], axis=0
                    )
                    all_gold[user_id] = np.concatenate(
                        [all_gold[user_id], user_gold], axis=0
                    )
                else:
                    all_predictions[user_id] = user_preds
                    all_gold[user_id] = user_gold
            
            if verbose and (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed batch {batch_idx + 1}: "
                           f"{len(all_predictions)} users so far")
    
    if verbose:
        logger.info(f"\nCollection complete:")
        logger.info(f"  Total users: {len(all_predictions)}")
        logger.info(f"  Total texts (valid): {total_texts}")
        logger.info(f"  Total padded positions (excluded): {total_padded}")
        logger.info(f"  Padding ratio: {total_padded / (total_texts + total_padded):.1%}")
        
        # Verify shapes
        for user_id, pred in all_predictions.items():
            gold = all_gold[user_id]
            assert pred.shape == gold.shape, f"Final shape mismatch for user {user_id}"
            assert pred.shape[1] == 2, f"Expected 2 dimensions (V, A), got {pred.shape[1]}"
    
    return all_predictions, all_gold

def collect_predictions_subtask2a(
    model,
    dataloader,
    device,
    verbose: bool = False
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Collect predictions for Task 2A (Forecasting).
    Enhanced version with validation and logging.
    """
    model.eval()
    
    all_preds = {}
    all_gold = {}
    
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move inputs to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            history_va = batch['history_va'].to(device)
            seq_lengths = batch['seq_lengths'].to(device)
            seq_mask = batch['seq_attention_mask'].to(device)
            
            # Gold targets 
            targets = batch['targets'].numpy() # [Batch, 2]
            user_ids = batch['user_ids']       # List[str]
            
            # Forward Pass
            preds = model(input_ids, attention_mask, history_va, seq_lengths, seq_mask)
            preds = preds.cpu().numpy()        # [Batch, 2]
            
            batch_size = preds.shape[0]
            total_samples += batch_size
            
            assert preds.shape == targets.shape, \
                f"Batch shape mismatch: preds {preds.shape} vs targets {targets.shape}"
            assert preds.shape[1] == 2, \
                f"Output dim mismatch: expected 2 (V,A), got {preds.shape[1]}"
            
            # Group by User
            for i, user_id in enumerate(user_ids):
                single_pred = preds[i]    # [2]
                single_gold = targets[i]  # [2]
                
                if user_id not in temp_preds:
                    all_preds[user_id] = []
                    all_gold[user_id] = []
                
                all_preds[user_id].append(single_pred)
                all_gold[user_id].append(single_gold)
            

    # Convert lists to numpy arrays
    final_preds = {u: np.array(v) for u, v in all_preds.items()}
    final_gold = {u: np.array(v) for u, v in all_gold.items()}
    
    if verbose:
        logger.info(f"\nCollection complete (Task 2A):")
        logger.info(f"  Total samples (windows): {total_samples}")
        logger.info(f"  Unique users: {len(final_preds)}")
        
        for uid in final_preds:
            p_shape = final_preds[uid].shape
            g_shape = final_gold[uid].shape
            assert p_shape == g_shape, f"User {uid} mismatch: {p_shape} vs {g_shape}"
    
    return final_preds, final_gold


def print_evaluation_results(results: Dict[str, float], title: str = "Evaluation Results"):
    """
    Pretty print evaluation results.
    
    Args:
        results: Dictionary from evaluate_subtask1()
        title: Title for the output
    """
    print("\n" + "=" * 60)
    print(f"SemEval 2026 EmoVA - {title}")
    print("=" * 60)
    
    print("\nVALENCE")
    print(f"  Between-user r:   {results['valence/r_between']:>7.4f}")
    print(f"  Within-user r:    {results['valence/r_within']:>7.4f}")
    print(f"  Composite r:      {results['valence/r_composite']:>7.4f}  ← ranking metric")
    print(f"  Between-user MAE: {results['valence/mae_between']:>7.4f}")
    print(f"  Within-user MAE:  {results['valence/mae_within']:>7.4f}")
    
    print("\nAROUSAL")
    print(f"  Between-user r:   {results['arousal/r_between']:>7.4f}")
    print(f"  Within-user r:    {results['arousal/r_within']:>7.4f}")
    print(f"  Composite r:      {results['arousal/r_composite']:>7.4f}  ← ranking metric")
    print(f"  Between-user MAE: {results['arousal/mae_between']:>7.4f}")
    print(f"  Within-user MAE:  {results['arousal/mae_within']:>7.4f}")
    
    print("\n" + "-" * 60)
    print(f"OVERALL COMPOSITE r: {results['overall/r_composite']:>7.4f}")
    print("=" * 60 + "\n")


def print_results_subtask2a(results: Dict[str, float], title: str = "Subtask 2A Results"):
    """
    Pretty print evaluation results for Subtask 2A (Forecasting).
    
    Args:
        results: Dictionary from evaluate_subtask2a()
        title: Title for the output
    """
    print("\n" + "=" * 60)
    print(f"SemEval 2026 EmoVA - {title}")
    print("=" * 60)
    
    print("\nVALENCE")
    print(f"  Avg Per-User r:   {results['valence/r_per_user']:>7.4f}")
    print(f"  Avg Per-User MAE: {results['valence/mae']:>7.4f}")
    
    print("\nAROUSAL")
    print(f"  Avg Per-User r:   {results['arousal/r_per_user']:>7.4f}")
    print(f"  Avg Per-User MAE: {results['arousal/mae']:>7.4f}")
    
    print("\n" + "-" * 60)
    # Nella Task 2, il ranking è dato dalla media delle correlazioni V e A
    print(f"OVERALL SCORE (Mean r): {results['overall/score']:>7.4f}  ← ranking metric")
    print("=" * 60 + "\n")

# =============================================================================
# Standalone Validation (for testing without model)
# =============================================================================

def validate_evaluation_inputs(
    predictions: Dict[str, np.ndarray],
    gold: Dict[str, np.ndarray]
) -> bool:
    """
    Validate that predictions and gold are properly formatted.
    
    Returns True if valid, raises AssertionError otherwise.
    """
    # Check user sets match
    pred_users = set(predictions.keys())
    gold_users = set(gold.keys())
    assert pred_users == gold_users, \
        f"User mismatch: {len(pred_users)} in predictions, {len(gold_users)} in gold"
    
    for user_id in predictions.keys():
        pred = predictions[user_id]
        g = gold[user_id]
        
        # Check shapes
        assert pred.ndim == 2, f"User {user_id}: predictions should be 2D, got {pred.ndim}D"
        assert g.ndim == 2, f"User {user_id}: gold should be 2D, got {g.ndim}D"
        assert pred.shape == g.shape, \
            f"User {user_id}: shape mismatch pred={pred.shape}, gold={g.shape}"
        assert pred.shape[1] == 2, \
            f"User {user_id}: expected 2 columns (V, A), got {pred.shape[1]}"
        
        # Check for NaN/Inf
        assert not np.isnan(pred).any(), f"User {user_id}: NaN in predictions"
        assert not np.isnan(g).any(), f"User {user_id}: NaN in gold"
        assert not np.isinf(pred).any(), f"User {user_id}: Inf in predictions"
        assert not np.isinf(g).any(), f"User {user_id}: Inf in gold"
        
        # Check value ranges (soft check - just warnings)
        if pred[:, 0].min() < -3 or pred[:, 0].max() > 3:
            logger.warning(f"User {user_id}: valence predictions outside [-3, 3]")
        if pred[:, 1].min() < -1 or pred[:, 1].max() > 3:
            logger.warning(f"User {user_id}: arousal predictions outside [-1, 3]")
    
    logger.info(f"Validation passed: {len(predictions)} users")
    return True


# =============================================================================
# Example Usage / Test
# =============================================================================

if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy data for testing
    np.random.seed(42)
    
    n_users = 20
    predictions = {}
    gold = {}
    
    for i in range(n_users):
        user_id = f"user_{i:03d}"
        n_texts = np.random.randint(5, 30)
        
        # Generate correlated predictions and gold
        gold_v = np.random.uniform(-1.5, 1.5, n_texts)
        gold_a = np.random.uniform(0.3, 1.7, n_texts)
        
        # Add noise to create predictions
        pred_v = gold_v + np.random.normal(0, 0.3, n_texts)
        pred_a = gold_a + np.random.normal(0, 0.2, n_texts)
        
        predictions[user_id] = np.stack([pred_v, pred_a], axis=-1)
        gold[user_id] = np.stack([gold_v, gold_a], axis=-1)
    
    # Validate inputs
    validate_evaluation_inputs(predictions, gold)
    
    # Run evaluation
    results = evaluate_subtask1(predictions, gold, verbose=True)
    
    # Print results
    print_evaluation_results(results, title="Test Evaluation")
