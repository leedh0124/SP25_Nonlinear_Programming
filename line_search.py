def Armijo_backtracking(f, x_k, f_xk, p_k, dpsi_0, alpha_init, c1, tau, *args):
    """
    Armijo backtracking line search for step size selection.
    Parameters:
    f : callable
        The objective function to minimize.
    x_k : array_like
        Current point.
    f_xk : float
        Function value at x_k.
    p_k : array_like
        Search direction.
    dpsi_0 : float
        Directional derivative at x_k.
    alpha_init : float
        Initial step size.
    c1 : float
        Parameter for sufficient decrease condition.
    tau : float
        Reduction factor for step size.
    Returns:
    alpha : float
        Step size that satisfies the Armijo condition.
    """
    alpha = alpha_init

    while f(x_k + alpha * p_k, *args) > f_xk + c1 * alpha * dpsi_0:
        alpha *= tau

    return alpha

