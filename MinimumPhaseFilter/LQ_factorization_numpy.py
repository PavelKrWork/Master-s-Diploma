import numpy as np
from scipy.linalg import qr

# QL factorization
def LQFactorization(H: np.array) -> tuple:
    H_transp = H.T
    Q_transp, R = qr(H_transp)
    L = R.T
    Q = Q_transp.T

    return L, Q