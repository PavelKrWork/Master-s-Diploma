import numpy as np
from LQ_factorization_numpy import LQFactorization

def test_QL():
    np.random.seed(42)
    M, N = 3, 3
    B = np.array([[1, 2, 3], [2, 4, 6], [4, 8, 12]])
    L, Q = LQFactorization(B)
    
    assert Q.shape == (M, M)
    assert L.shape == (M, N)

    assert np.allclose(B, L @ Q, atol=1e-10)

    assert np.allclose(Q @ Q.T, np.eye(M), atol=1e-10)
    assert np.allclose(Q.T @ Q, np.eye(M), atol=1e-10)

if __name__ == "__main__":
    test_QL()