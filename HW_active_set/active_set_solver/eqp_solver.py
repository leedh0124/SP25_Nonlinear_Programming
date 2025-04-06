import numpy as np
from scipy.linalg import ldl, solve_triangular

def solve_ldl(L, D, perm, rhs):
    """
    Solve the system K x = rhs given the LDLᵀ factorization from scipy.linalg.ldl:
         K[perm, :][:, perm] = L @ D @ Lᵀ.
    
    This routine uses built-in functions for:
      - Forward substitution: solve L y = rhs_perm using solve_triangular.
      - Solving the diagonal/block-diagonal system: D z = y via np.linalg.solve.
      - Backward substitution: solve Lᵀ w = z using solve_triangular.
    
    Finally, it undoes the permutation to return x.
    
    Parameters:
        L    : Unit lower triangular matrix from the LDLᵀ factorization.
        D    : Block-diagonal matrix (with possible 1x1 and 2x2 blocks).
        perm : Permutation vector such that K[perm, :][:, perm] = L D Lᵀ.
        rhs  : Right-hand side vector.
    
    Returns:
        x    : Solution vector of the system K x = rhs.
    """
    # Permute the right-hand side according to the factorization ordering.
    rhs_perm = rhs[perm]
    
    # Solve L y = rhs_perm (forward substitution)
    y = solve_triangular(L, rhs_perm, lower=True, unit_diagonal=True)
    
    # Solve D z = y using the built-in solver (handles block-diagonal structure)
    z = np.linalg.solve(D, y)
    
    # Solve Lᵀ w = z (backward substitution)
    w = solve_triangular(L.T, z, lower=False, unit_diagonal=True)
    
    # Undo the permutation to get the solution for the original ordering.
    inv_perm = np.argsort(perm)
    x = np.zeros_like(w)
    x[inv_perm] = w
    return x

def solve_eqp(G, c, A, b):
    """
    Solve the KKT system for the equality-constrained QP:
    
         minimize    0.5 * xᵀ G x + cᵀ x
         subject to  A x = b
    
    The KKT conditions lead to the system:
    
         [ G   Aᵀ ] [ x  ] = [ -c ]
         [ A    0 ] [ -λ ]   [  b ]
    
    This function builds the KKT matrix, factorizes it via LDLᵀ factorization 
    using scipy.linalg.ldl, and then solves the system using the built-in routines.
    
    Returns:
        primal_x   : Primal solution vector.
        dual_lambda : Lagrange multipliers (dual solution).
    """
    n = G.shape[0] # number of variables
    m = A.shape[0] # number of equality constraints
    
    # Form the KKT matrix.
    K_top = np.hstack((G, A.T))
    K_bottom = np.hstack((A, np.zeros((m, m))))
    K = np.vstack((K_top, K_bottom))
    
    # Build the right-hand side vector [ -c ; b ].
    rhs = np.concatenate((-c, b))
    
    # Compute the LDLᵀ factorization.
    # K[perm, :][:, perm] = L @ D @ Lᵀ.
    L, D, perm = ldl(K, lower=True)
    
    # Solve the system using the built-in solve_ldl function.
    sol = solve_ldl(L, D, perm, rhs)
    
    # Extract the primal and dual variables.
    primal_x = sol[:n]
    dual_lambda = -sol[n:]
    return primal_x, dual_lambda
