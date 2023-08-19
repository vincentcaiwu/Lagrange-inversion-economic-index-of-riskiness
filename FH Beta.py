import math
import numpy as np
import sympy as sp
import pandas as pd
from scipy.stats import beta

def riskiness_beta(m, L=100, a=2, b=2):
    x = sp.Symbol('x')
    r = sp.Symbol('r', positive=True)
    integrand = sp.log(1 + x / r) * (x + L) * (m - x)
    result = sp.integrate(integrand, (x, -L, m))
    solution = sp.nsolve(result, 101)
    return solution

def lagrange_inversion_log_beta(m, L=100, iter=100, a=2, b=2):
    mean = beta.moment(1, a, b, loc=-L, scale=m+L)
    a1 = beta.moment(2, a, b, loc=-L, scale=m+L) / 2
    series = mean / a1
    A_hat = []
    for n in range(2, iter+1):
        series_old = series
        a_n1 = (-1)**(n + 1) * beta.moment(n + 1, a, b, loc=-L, scale=m+L) / (n + 1) * math.factorial(n)
        a_n_hat = a_n1 / (n * a1)
        A_hat.append(a_n_hat)
        bn = 0
        for k in range(1, n):
            bn += (-1)**k * math.factorial(n + k - 1) / math.factorial(n - 1) * sp.bell(n - 1, k, A_hat[:(n - k)])
        series += bn * mean**n / math.factorial(n) / a1**n
        if abs(1 / series - 1 / series_old) <= 1e-3:
            break
    return 1 / series

M = np.arange(101, 121, 1)
riskiness_explicit = [riskiness_beta(m) for m in M]
riskiness_approx = [lagrange_inversion_log_beta(m) for m in M]

Error = np.round(np.array((np.array(riskiness_approx) - np.array(riskiness_explicit)) / riskiness_explicit * 100).astype(float), 4)
riskiness_approx = np.round(np.array(riskiness_approx).astype(float), 4)
riskiness_explicit = np.round(np.array(riskiness_explicit).astype(float), 4)
df = pd.DataFrame({
    "m": M, 
    "Approximate riskiness": riskiness_approx, 
    "Explicit riskiness": riskiness_explicit, 
    "Approximation %error": Error
    })
print(df)