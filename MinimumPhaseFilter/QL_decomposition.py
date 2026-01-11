import numpy as np

def QL_factorize(B: np.array) -> tuple:
    B = np.array(B, dtype=complex)
    M, N = B.shape
    K = min(M, N)
    
    B_hat = np.copy(B)
    
    # Список для хранения матриц U_k
    U_list = []
    
    for k in range(1, K + 1):
        # Выделяем последний столбец B_hat
        b = B_hat[:, -1]
        
        k_tilde = M - k + 1
        k_tilde_idx = k_tilde - 1
        
        alpha = np.linalg.norm(b)
        alpha_tilde = np.exp(1j * np.angle(b[k_tilde_idx])) * alpha
        e_k = np.zeros(len(b), dtype=complex)
        e_k[k_tilde_idx] = 1
        v = b + alpha_tilde * e_k
        
        # Строим матрицу отражения Хаусхолдера
        v_norm_sq = np.linalg.norm(v)**2
        if v_norm_sq != 0:
            U_k_small = np.exp(-1j * np.angle(b[k_tilde_idx])) * (
                2 * np.outer(v, v.conj()) / v_norm_sq - np.eye(len(b))
            )
        else:
            U_k_small = np.eye(len(b), dtype=complex)
        
        B_hat = U_k_small @ B_hat
        
        # Расширяем U_k_small до полной матрицы MxM для сохранения
        U_k_full = np.eye(M, dtype=complex)
        start_idx = M - len(b)
        U_k_full[start_idx:, start_idx:] = U_k_small
        
        U_list.append(U_k_full)
        
        if k < K:
            B_hat = B_hat[:-1, :-1]
    
    L = np.copy(B)
    for U_k in U_list:
        L = U_k @ L
    
    Q = np.eye(M, dtype=complex)
    for U_k in reversed(U_list):
        Q = U_k.conj().T @ Q
    
    return L, Q
