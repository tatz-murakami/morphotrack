import numpy as np


def random_idx_with_max(N,M):
    if M == -1 or M >= N:
        idx = np.arange(N)
    else:
        idx = np.random.choice(N, size=M, replace=False)
    return idx


