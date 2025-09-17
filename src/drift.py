import numpy as np
from scipy.stats import ks_2samp

def psi(expected, actual, buckets=10):
    expected = expected[~np.isnan(expected)]; actual = actual[~np.isnan(actual)]
    if expected.size == 0 or actual.size == 0: return float('nan')
    bins = np.percentile(expected, np.linspace(0,100,buckets+1)); bins[0]=-np.inf; bins[-1]=np.inf
    e_counts,_ = np.histogram(expected, bins=bins); a_counts,_ = np.histogram(actual, bins=bins)
    e_prop = e_counts / max(e_counts.sum(),1); a_prop = a_counts / max(a_counts.sum(),1)
    e_prop = np.where(e_prop==0,1e-6,e_prop); a_prop = np.where(a_prop==0,1e-6,a_prop)
    return float(np.sum((a_prop-e_prop)*np.log(a_prop/e_prop)))

def ks_stat(expected, actual):
    expected = expected[~np.isnan(expected)]; actual = actual[~np.isnan(actual)]
    if expected.size == 0 or actual.size == 0: return float('nan')
    return float(ks_2samp(expected, actual).statistic)
