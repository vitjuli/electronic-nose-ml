import numpy as np

def residual_calibration_intervals(y_true_val, y_pred_val, y_pred_test, alpha=0.1):
    res = np.abs(y_true_val - y_pred_val)
    q = np.quantile(res, 1 - alpha, axis=0)
    return y_pred_test - q, y_pred_test + q
