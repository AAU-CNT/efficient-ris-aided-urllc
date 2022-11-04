# eval_cdf_rice.py: estimate the CDF of the double rice
from os import path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rice

from environment import DATADIR
from scenario.common import printplot, db2lin


def ecdf(a):
    x, counts = np.unique(a, return_counts=True)
    cusum = np.cumsum(counts)
    return x, cusum / cusum[-1]


# param
confidence = 0.98
epsilon = 1e-4
num_samples = int(np.round(np.log(2 / confidence) / 2 / epsilon ** 2))
K_vec = np.arange(6, 7, 1)   # dB
x = np.zeros((num_samples, len(K_vec)))
CDF = np.zeros((num_samples, len(K_vec)))
# G0 = np.zeros(K_vec.shape)
# G0_2 = np.zeros(K_vec.shape)

for i, K in enumerate(K_vec):
    # Compute parameters
    K_lin = db2lin(K)
    nu = np.sqrt(K_lin / (1 + K_lin))
    sigma = np.sqrt(1 / 2 / (K_lin + 1))

    # distribution and sampling
    gm = rice.rvs(nu / sigma, scale=sigma, size=num_samples)
    gu = rice.rvs(nu / sigma, scale=sigma, size=num_samples)

    g = np.abs(gm * gu) ** 2

    # Compute the empirical CDF
    x[:, i], CDF[:, i] = ecdf(g)

    plt.plot(x[:, i], CDF[:, i], label=f'$K = {K}$ dB')

plt.xlim(0, 5)
plt.ylim(0, 1)
printplot(labels=[r'$x$', r'$C_R(x)$'])

np.save(path.join(DATADIR, 'CDF'), (x, CDF))

