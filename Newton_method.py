import numpy as np
from line_search import Armijo_backtracking

def Newton_method(f, grad_f, hess_f, x0, alpha_init=1, c1=1e-4, tau=0.5, tol=1e-6, max_iter=1000, *args):
    """
    Newton's Method for optimization.
    Parameters:
    f : callable
        The objective function to minimize.
    grad_f : callable
        The gradient of the objective function.
    hess_f : callable
        The Hessian of the objective function.
    x0 : array_like
        Initial guess for the variables.
    alpha_init : float
        Initial step size.
    c1 : float
        Parameter for sufficient decrease condition.
    tau : float
        Reduction factor for step size.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    *args : tuple
        Additional arguments to pass to the objective function and its gradient.
    Returns:
    x_k : array_like
        The point that minimizes the objective function.
    f_val : float
        The value of the objective function at the minimum point.
    k : int
        Number of iterations performed.
    """
    # print(f"{'Iter':>4}  {'f':>10}  {'||grad||':>10}  {'alpha':>10}")
    x_k = np.array(x0, dtype=float)
    f_xk = f(x_k, *args)
    grad_xk = grad_f(x_k, *args)
    norm_grad_x0 = np.linalg.norm(grad_xk)
    hess_xk = hess_f(x_k, *args)
    diff_x = 1e10
    diff_f = 1e10
    k = 0

    while k < max_iter and np.linalg.norm(grad_xk) > tol * max(1, norm_grad_x0): #   
        p_k = np.linalg.solve(hess_xk, -grad_xk)
        dpsi_0 = grad_xk.T @ p_k

        alpha_k = Armijo_backtracking(f, x_k, f_xk, p_k, dpsi_0, alpha_init, c1, tau, *args)

        x_kp1 = x_k + alpha_k * p_k

        diff_f = (f(x_kp1, *args) - f_xk)
        alpha_init = 2 * diff_f / dpsi_0

        diff_x = x_kp1 - x_k
        x_k = x_kp1

        f_xk = f(x_k, *args)
        grad_xk = grad_f(x_k, *args)
        hess_xk = hess_f(x_k, *args)

        k += 1

        # print(f"{k:4d}  {f_xk:10.2e}  {np.linalg.norm(grad_xk):10.2e}  {alpha_k:10.2e}")
        
    print(f"Converged in {k} iterations.")

    return x_k, f_xk, k

def Newton_modified_Cholesky_Added_Multiple_Identity(f, grad_f, hess_f, x0, alpha_init=1, c1=1e-4, tau=0.5, beta=1e-4, tol=1e-6, max_iter=1000, *args):
    """
    Modified Newton's Method for optimization.
    Parameters:
    f : callable
        The objective function to minimize.
    grad_f : callable
        The gradient of the objective function.
    hess_f : callable
        The Hessian of the objective function.
    x0 : array_like
        Initial guess for the variables.
    alpha_init : float
        Initial step size.
    c1 : float
        Parameter for sufficient decrease condition.
    tau : float
        Reduction factor for step size.
    beta : float
        Small positive number to add to the diagonal of the Hessian.    
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    *args : tuple
        Additional arguments to pass to the objective function and its gradient.
    Returns:
    x_k : array_like
        The point that minimizes the objective function.
    f_val : float
        The value of the objective function at the minimum point.
    k : int
        Number of iterations performed.
    """
    # print(f"{'Iter':>4}  {'f':>10}  {'||grad||':>10}  {'alpha':>10}")
    x_k = np.array(x0, dtype=float)
    f_xk = f(x_k, *args)
    grad_xk = grad_f(x_k, *args)
    norm_grad_x0 = np.linalg.norm(grad_xk)
    hess_xk = hess_f(x_k, *args)
    diff_x = 1e10
    diff_f = 1e10
    k = 0

    while k < max_iter and np.linalg.norm(grad_xk) > tol * max(1, norm_grad_x0): #  and np.linalg.norm(diff_f) > tol
        L_k = Cholesky_with_multiple_of_identity(hess_xk, beta)
        B_k = L_k @ L_k.T
        p_k = np.linalg.solve(B_k, -grad_xk)
        dpsi_0 = grad_xk.T @ p_k

        alpha_k = Armijo_backtracking(f, x_k, f_xk, p_k, dpsi_0, alpha_init, c1, tau, *args)

        x_kp1 = x_k + alpha_k * p_k

        diff_f = (f(x_kp1, *args) - f_xk)
        alpha_init = 2 * diff_f / dpsi_0

        diff_x = x_kp1 - x_k
        x_k = x_kp1

        f_xk = f(x_k, *args)
        grad_xk = grad_f(x_k, *args)
        hess_xk = hess_f(x_k, *args)

        k += 1

        # print(f"{k:4d}  {f_xk:10.2e}  {np.linalg.norm(grad_xk):10.2e}  {alpha_k:10.2e}")

    print(f"Converged in {k} iterations.")

    return x_k, f_xk, k

def Cholesky_with_multiple_of_identity(A, beta):
    """
    Cholesky decomposition with a multiple of the identity matrix added.
    Parameters:
    A : array_like
        The matrix to decompose.
    beta : float
        small positive number to add to the diagonal.
    Returns:
    L : array_like
        The lower triangular matrix from the Cholesky decomposition.
    """
    if np.min(np.diag(A)) > 0:
        delta = 0
    else:
        delta = np.abs(np.min(np.diag(A))) + beta

    while True:
        try:
            L = np.linalg.cholesky(A + delta * np.eye(A.shape[0]))
            break

        except np.linalg.LinAlgError:
            delta = np.max([2 * delta, beta])

    return L
