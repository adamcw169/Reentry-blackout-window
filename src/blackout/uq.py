"""Quasi-Monte Carlo sampling and Sobol sensitivity indices."""

import numpy as np
from scipy.stats import qmc


def sobol_points(n, d, seed=0, scramble=True):
    """n scrambled Sobol points in [0,1)^d (n should be a power of 2)."""
    return qmc.Sobol(d, scramble=scramble, seed=seed).random(n)


def random_points(n, d, seed=0):
    return np.random.default_rng(seed).random((n, d))


def saltelli_matrices(n, d, seed=0):
    """A, B and the d mixed matrices AB_i (column i of A replaced by B), from a 2d-dim Sobol set."""
    base = sobol_points(n, 2 * d, seed=seed)
    A, B = base[:, :d], base[:, d:]
    AB = np.repeat(A[None, :, :], d, axis=0)
    for i in range(d):
        AB[i, :, i] = B[:, i]
    return A, B, AB


def sobol_indices(fA, fB, fAB):
    """First-order (Saltelli 2010) and total (Jansen) indices.

    fA, fB : (n,) model outputs; fAB : (d, n)
    """
    var = np.var(np.concatenate([fA, fB]), ddof=1)
    S1 = np.mean(fB[None, :] * (fAB - fA[None, :]), axis=1) / var
    ST = 0.5 * np.mean((fA[None, :] - fAB) ** 2, axis=1) / var
    return S1, ST


def bootstrap_ci(fA, fB, fAB, n_boot=200, seed=1):
    rng = np.random.default_rng(seed)
    n = len(fA)
    S1s, STs = [], []
    for _ in range(n_boot):
        k = rng.integers(0, n, n)
        a, b = sobol_indices(fA[k], fB[k], fAB[:, k])
        S1s.append(a)
        STs.append(b)
    q = lambda x: np.percentile(np.array(x), [5, 95], axis=0)
    return q(S1s), q(STs)
