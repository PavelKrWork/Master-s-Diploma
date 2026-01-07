import numpy as np

def HouseHolderQL(B: np.array, is_complex: True) -> tuple:
    el_type = complex if is_complex else float
        
    M, N = B.shape
    Q = np.eye(M, dtype=el_type)
    L = B.astype(el_type)
    
    for i in range(N-1, -1, -1):
        b = L[:, i]
        k = M - i

        if is_complex:
            ang = np.angle(b[-1])
            alpha = np.linalg.norm(b)
            alpha_tilde = np.exp(1j * ang) * alpha
            
            e = np.zeros(M, dtype=el_type)
            e[k - 1] = 1
            v = b + alpha_tilde * e
            
            U_k = np.exp(-1j * ang) * (2 * np.outer(v, v.conj()) / np.linalg.norm(v)**2 - np.eye(M))

        else:
            e = np.zeros(M, dtype=el_type)
            e[-1] = 1
            alpha = np.linalg.norm(b)
            v = b + alpha * e
            
            U_k = 2 * np.outer(v, v) / np.linalg.norm(v)**2 - np.eye(M)
            
        L = U_k @ L
        Q = Q @ U_k.conj().T if is_complex else Q @ U_k.T
    
    return Q, L