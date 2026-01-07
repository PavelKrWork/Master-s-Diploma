# import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from MinimumPhaseFilter.QL_decomposition import HouseHolderQL

def test_QL():
    np.random.seed(42)
    M, N = 3, 3
    B = np.array([[1, 2, 3], [2, 4, 6], [4, 8, 12]])
    Q, L = HouseHolderQL(B, is_complex=True)

    print("Q = ", Q)
    print()
    print("L = ", L)
    
    # Проверка размеров
    assert Q.shape == (M, M)
    assert L.shape == (M, N)

    # Проверка: B = Q * L
    assert np.allclose(B, Q @ L, atol=1e-10)

    # Проверка, что Q - ортогональная 
    assert np.allclose(Q @ Q.T, np.eye(M), atol=1e-10)
    assert np.allclose(Q.T @ Q, np.eye(M), atol=1e-10)

if __name__ == "__main__":
    test_QL()