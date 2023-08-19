import math
import numpy as np
import sympy as sp
import pandas as pd
from scipy.stats import gamma
from scipy.special import lambertw

def riskiness_gamma(a, b=8, s=1000):
    if s < a * b:
        return (1 / (-1 / b - a / s * lambertw(-s * np.exp(-s / a / b) / a / b, -1))).real
    else:
        return np.inf

def lagrange_inversion_log_gamma(a, b=8, s=1000, iter=100):
    mean = gamma.moment(1, a, loc=-s, scale=b)
    a1 = gamma.moment(2, a, loc=-s, scale=b) / 2
    series = mean / a1
    A_hat = []
    for n in range(2, iter+1):
        series_old = series
        a_n1 = (-1)**(n + 1) * gamma.moment(n + 1, a, loc=-s, scale=b) / (n + 1)
        a_n_hat = a_n1 / (n * a1)
        A_hat.append(a_n_hat)
        bn = 0
        for k in range(1, n):
            bn += (-1)**k * math.factorial(n + k - 1) / math.factorial(n - 1) * sp.bell(n - 1, k, A_hat[:(n - k)])
        series += bn * mean**n / math.factorial(n) / a1**n
        if abs(1/series - 1/series_old) <= 1e-2:
            break
    return 1 / series

A = np.arange(125.1, 127.1, 0.1)
riskiness_explicit = [riskiness_gamma(a) for a in A]
riskiness_approx = [lagrange_inversion_log_gamma(a) for a in A]

Error = np.round(np.array((np.array(riskiness_approx) - np.array(riskiness_explicit)) / riskiness_explicit * 100).astype(float), 4)
riskiness_approx = np.round(np.array(riskiness_approx).astype(float), 4)
riskiness_explicit = np.round(np.array(riskiness_explicit).astype(float), 4)
df = pd.DataFrame({
    "a": A, 
    "Approximate riskiness": riskiness_approx, 
    "Explicit riskiness": riskiness_explicit, 
    "Approximation %error": Error
    })
print(df)