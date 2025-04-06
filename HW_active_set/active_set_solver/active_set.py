import numpy as np
from eqp_solver import solve_eqp

def objective(x, G, c):
    """Compute the quadratic objective: 0.5*xᵀGx + cᵀx."""
    return 0.5 * x.T @ G @ x + c.T @ x

def active_set_qp(G, c, A, b, x0, tol=1e-8, max_iter=50):
    """
    Solve the convex QP
         minimize    0.5*xᵀ G x + cᵀ x
         subject to  A x ≥ b
    using an active‑set method.
        
    Parameters:
      G       : (n x n) symmetric positive definite matrix.
      c       : (n,) vector.
      A       : (m x n) matrix, where each row aᵢᵀ defines an inequality constraint aᵢᵀ*x ≥ bᵢ.
      b       : (m,) vector.
      x0      : Feasible starting point (must satisfy A*x ≥ b).
      tol     : Tolerance for determining activity and convergence.
      max_iter: Maximum number of iterations.
      
    Returns:
      x_opt     : Optimal solution.
      lambda_opt: Dictionary mapping indices of active inequality constraints to multipliers.
      history   : List of tuples (working set, f(x), α, ||p||², x_iterate) for each iteration.
    """
    x = x0.copy()
    n_constraints = A.shape[0]
    
    # Check feasibility of the initial point.
    for i in range(n_constraints):
        if A[i] @ x < b[i] - tol:
            raise ValueError(f"Initial point is infeasible: Constraint {i} violated, A[i]x = {A[i]@x}, b[i]={b[i]}")
    
    # Initialize working set with indices of constraints active at x0.
    W = []
    for i in range(n_constraints):
        if np.abs(A[i] @ x - b[i]) < tol:
            W.append(i)
    
    history = []
    
    for k in range(max_iter):
        f_val = objective(x, G, c)
        
        # Form the working-set matrix A_W.
        if len(W) > 0:
            A_W = A[W, :]
        else:
            A_W = np.empty((0, x.size))
        
        # Solve the equality-constrained QP subproblem:
        #   min_p 0.5*pᵀGp + pᵀ*(Gx+c)   subject to   A_W p = 0.
        if A_W.shape[0] > 0:
            p, lambda_W = solve_eqp(G, G @ x + c, A_W, np.zeros(A_W.shape[0]))
        else:
            # Unconstrained step.
            p = - np.linalg.solve(G, (G @ x + c))
            lambda_W = np.array([])
        
        p_norm = np.linalg.norm(p)
        
        if p_norm < tol:
            # No descent direction: check Lagrange multipliers.
            if len(W) == 0:
                history.append((W.copy(), f_val, None, p_norm**2, x.copy()))
                return x, {}, history
            if all(lambda_W >= -tol):
                history.append((W.copy(), f_val, None, p_norm**2, x.copy()))
                lambda_dict = {i: lambda_W[j] for j, i in enumerate(W)}
                return x, lambda_dict, history
            else:
                # Remove the constraint with the most negative multiplier.
                lambda_W = np.array(lambda_W)
                j_index = np.argmin(lambda_W)
                j = W[j_index]
                print(f"Iteration {k}: p≈0 but lambda[{j}] = {lambda_W[j_index]:.4f} < 0; removing constraint {j}.")
                W.pop(j_index)
                history.append((W.copy(), f_val, 0, p_norm**2, x.copy()))
        else:
            # Compute step length α.
            # For constraints A*x ≥ b, a constraint may become violated if p decreases aᵢᵀx.
            # That is, only if aᵢᵀp < -tol does the constraint become blocking.
            alpha = 1.0
            blocking_constraint = None
            for i in range(n_constraints):
                if i in W:
                    continue
                a_i = A[i]
                if a_i @ p < -tol:
                    alpha_i = (a_i @ x - b[i]) / (- (a_i @ p))
                    if alpha_i < alpha:
                        alpha = alpha_i
                        blocking_constraint = i
            x = x + alpha * p
            if blocking_constraint is not None:
                W.append(blocking_constraint)
                print(f"Iteration {k}: Step length α = {alpha:.4f} with blocking constraint {blocking_constraint} added.")
            else:
                print(f"Iteration {k}: Full step taken (α = {alpha:.4f}).")
            history.append((W.copy(), objective(x, G, c), alpha, p_norm**2, x.copy()))
        
        print(f"Iteration {k}: Working set = {W}, f(x) = {f_val:.4f}, α = {'N/A' if p_norm < tol else f'{alpha:.4f}'}, ||p||² = {p_norm**2:.4e}, x = {x}")
    
    print("Maximum iterations reached.")
    return x, {}, history
