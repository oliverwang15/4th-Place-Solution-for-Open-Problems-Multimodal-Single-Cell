import torch
import numpy as np
import pandas as pd

def partial_correlation_score_torch_faster(y_true, y_pred):
    """Compute the correlation between each rows of the y_true and y_pred tensors.
    Compatible with backpropagation.
    """
    if type(y_true) == np.ndarray: y_true = torch.tensor(y_true)
    if type(y_pred) == np.ndarray: y_pred = torch.tensor(y_pred)

    y_true_centered = y_true - torch.mean(y_true, dim=1)[:,None]
    y_pred_centered = y_pred - torch.mean(y_pred, dim=1)[:,None]
    cov_tp = torch.sum(y_true_centered*y_pred_centered, dim=1)/(y_true.shape[1]-1)
    var_t = torch.sum(y_true_centered**2, dim=1)/(y_true.shape[1]-1)
    var_p = torch.sum(y_pred_centered**2, dim=1)/(y_true.shape[1]-1)
    return cov_tp/torch.sqrt(var_t*var_p)

def correl_loss(pred, tgt):
    """Loss for directly optimizing the correlation.
    """
    return -torch.mean(partial_correlation_score_torch_faster(tgt, pred))

def aMAPE_loss(pred, tgt):
    assert pred.shape == tgt.shape
    ones = torch.ones_like(tgt)
    zeros = torch.zeros_like(tgt)
    zero_mask = torch.where(tgt == 0 ,ones,zeros)
    non_zero_mask = torch.where(tgt == 0 ,zeros,ones)

    # process non_zero with mape
    mape_loss = tgt + zero_mask
    mape_loss = torch.abs((pred-mape_loss)/mape_loss)*100
    mape_loss *= non_zero_mask
    mape_loss = torch.mean(mape_loss)
    # process zero with mse
    mse_loss = torch.nn.functional.mse_loss(pred,tgt)
    # mse_loss *= zero_mask
    loss_all = mape_loss+mse_loss

    return  loss_all


def correlation_score(y_true, y_pred):
    """Scores the predictions according to the competition rules. 
    
    It is assumed that the predictions are not constant.
    
    Returns the average of each sample's Pearson correlation coefficient"""
    if type(y_true) == pd.DataFrame: y_true = y_true.values
    if type(y_pred) == pd.DataFrame: y_pred = y_pred.values
    if y_true.shape != y_pred.shape: raise ValueError("Shapes are different.")
    corrsum = 0
    for i in range(len(y_true)):
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corrsum / len(y_true)