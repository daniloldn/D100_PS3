import numpy as np
import pandas as pd


def evaluate_pred(predictions: np.array, actual: np.array, weights: np.array)-> pd.DataFrame:

    #validating inputs
    y_pred = np.asarray(predictions, dtype=float)
    y_true = np.asarray(actual, dtype=float)
    weight = np.asarray(weights, dtype = float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    n = y_true.size

    if weight is None:
        weight = np.ones(n, dtype=float)
    else:
        weight = np.asarray(weight, dtype=float)
        if weight.shape != y_true.shape:
            raise ValueError("weight must have the same shape as y_true/y_pred")
        if np.any(weight < 0):
            raise ValueError("weight (sample weights) must be non-negative")
        
    total_weight = weight.sum()
    if total_weight == 0:
        raise ValueError("sum of weight is zero; cannot compute weight-weighted metrics")
    
    #bias calculations 

    residual = y_pred - y_true

    # Weight-weighted bias 
    bias_weight_weighted = np.sum(residual * weight) / total_weight

    #deviance 
    deviance = np.sum((residual**2) * weight) / total_weight

    #RMSE
    rmse = np.sqrt(deviance)
    #MAE
    mae = np.sum(np.absolute(residual)*weight)/total_weight

    df = pd.DataFrame({
        'bias_weight_weighted': [bias_weight_weighted],
        'deviance': [deviance], 
        'mae': [mae],
        'rmse': [rmse]
    })

    return df



