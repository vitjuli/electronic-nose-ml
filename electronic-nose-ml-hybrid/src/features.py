import numpy as np
from scipy.stats import skew, kurtosis

def features_from_cycle(x, prefix='sig'):
    d = {}
    d[f'{prefix}_mean']=float(x.mean()); d[f'{prefix}_std']=float(x.std()); d[f'{prefix}_min']=float(x.min()); d[f'{prefix}_max']=float(x.max()); d[f'{prefix}_ptp']=float(np.ptp(x))
    d[f'{prefix}_skew']=float(skew(x)); d[f'{prefix}_kurt']=float(kurtosis(x))
    dx = np.diff(x); d[f'{prefix}_d_mean']=float(dx.mean()); d[f'{prefix}_d_std']=float(dx.std())
    peaks = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0]; d[f'{prefix}_peaks']=int(peaks.size)
    return d
