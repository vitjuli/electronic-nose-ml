import torch

def pinball_loss(y_pred, y_true, quantiles):
    B, Ty = y_true.shape; Q = len(quantiles)
    assert y_pred.shape[1] == Ty*Q
    loss = 0.0
    for qi, q in enumerate(quantiles):
        yq = y_pred[:, qi*Ty:(qi+1)*Ty]
        e = y_true - yq
        loss += torch.maximum(q*e, (q-1)*e).mean()
    return loss / Q
