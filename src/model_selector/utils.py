import numpy as np
import pandas as pd

def calc_ks_score(y_true, y_prob):
    """
    Calculate the Kolmogorov-Smirnov (KS) statistic for binary classification.
    
    The KS statistic measures the maximum difference between the cumulative distribution
    functions of positive and negative classes. It's a common metric in credit scoring
    and binary classification problems.
    
    Args:
        y_true (array-like): True binary labels.
        y_prob (array-like): Predicted probabilities for the positive class.
        
    Returns:
        float: The KS statistic value, ranging from 0 to 1. Higher values indicate
              better separation between positive and negative classes.
    """
    data = pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).sort_values(
        "y_prob", ascending=False
    )
    data["cum_pos"] = (data["y_true"] == 1).cumsum()
    data["cum_neg"] = (data["y_true"] == 0).cumsum()
    total_pos = data["y_true"].sum()
    total_neg = (data["y_true"] == 0).sum()
    data["cum_pos_rate"] = data["cum_pos"] / total_pos
    data["cum_neg_rate"] = data["cum_neg"] / total_neg
    data["ks"] = data["cum_pos_rate"] - data["cum_neg_rate"]
    return data["ks"].max()

def blend_predictions(probs_list, weights=None):
    """
    Blend multiple model predictions using weighted averaging.
    
    This function combines predictions from multiple models into a single prediction
    by taking a weighted average. If no weights are provided, equal weighting is used.
    
    Args:
        probs_list (list): List of arrays, each containing predicted probabilities
                          from a different model. All arrays must have the same shape.
        weights (list, optional): List of weights for each model. Defaults to None,
                                 which assigns equal weights to all models.
    
    Returns:
        numpy.ndarray: Blended predictions with the same shape as each input array.
    """
    if weights is None:
        weights = [1.0 / len(probs_list)] * len(probs_list)
    final_prob = np.zeros_like(probs_list[0])
    for w, p in zip(weights, probs_list):
        final_prob += w * p
    return final_prob
