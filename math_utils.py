import numpy as np
from math import radians, sin, cos, sqrt, asin, fabs

def cosSim(v1: "ndarray", v2: "ndarray") -> "ndarray":
    """
    コサイン類似度を算出
    v1 : shape = (N,)
    v2 : shape = (N,)
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def standardization(x, axis=None, ddof=0):
    alpha = 1e-10
    x_mean = x.mean(axis=axis, keepdims=True)
    x_std = x.std(axis=axis, keepdims=True, ddof=ddof)

    return (x - x_mean) / (x_std + alpha)


def normalization(x, amin=0, amax=1):
    xmax = x.max()
    xmin = x.min()

    if xmin == xmax:
        return np.ones_like(x)

    return (amax - amin) * (x - xmin) / (xmax - xmin) + amin

def gini(y):
    m = statistics.mean(y)
    n = len(y)
    a = 2 * m * (n * (n - 1))
    ysum = 0

    for i in range(n):
        for j in range(n):
            ysum = ysum + (fabs(y[i] - y[j]))

    return(ysum / a)
