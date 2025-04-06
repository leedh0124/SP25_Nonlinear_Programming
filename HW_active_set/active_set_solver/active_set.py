import numpy as np
from eqp_solver import solve_eqp

def objective(x, G, c):
    """Compute the quadratic objective: 0.5*xᵀGx + cᵀx."""
    return 0.5 * x.T @ G @ x + c.T @ x

def active_set_qp(G, c, A, b, C, d, x0, tol=1e-8, max_iter=50):
    """
    Solve the convex QP
         minimize    0.5*xᵀ G x + cᵀ x
         subject to  A x ≥ b,   C x = d
    using an active‑set method.
    
    Parameters:
      G       : (n x n) symmetric positive definite matrix.
      c       : (n,) vector.
      A       : (m x n) matrix for inequality constraints (A x ≥ b). May be empty.
      b       : (m,) vector.
      C       : (p x n) matrix for equality constraints (C x = d). May be empty.
      d       : (p,) vector.
      x0      : Feasible starting point (satisfies A x ≥ b and C x = d).
      tol     : Tolerance for determining activity and convergence.
      max_iter: Maximum number of iterations.
      
    Returns:
      x_opt     : Optimal solution.
      lambda_opt: Dictionary mapping indices of active inequality constraints to multipliers.
                  (Multipliers corresponding to equality constraints are not used for removal.)
      history   : List of tuples (working set, f(x), α, ||p||²) for each iteration.
    """
    x = x0.copy()
    n_constraints = A.shape[0] if A.size > 0 else 0 # inequality constraints
    
    # Check feasibility of the initial point.
    for i in range(n_constraints):
        if A[i] @ x < b[i] - tol:
            raise ValueError(f"Initial point is infeasible: Constraint {i} violated, A[i]x = {A[i]@x}, b[i]={b[i]}")
    
    # Check feasibility for equality constraints.
    if C.size > 0:
        p_eq = C.shape[0]
        for i in range(p_eq):
            if abs(C[i] @ x - d[i]) > tol:
                raise ValueError(f"Infeasible starting point: Equality {i} violated, C[i]x = {C[i]@x}, d[i] = {d[i]}")
    else:
        p_eq = 0
        
    # Initialize working set with indices of constraints active at x0.
    W = []
    for i in range(n_constraints):
        if np.abs(A[i] @ x - b[i]) < tol:
            W.append(i)
    
    history = []
    
    for k in range(max_iter):
        f_val = objective(x, G, c)
        
        # Form the combined equality constraint matrix for the subproblem.
        # Always include equality constraints, and add the active inequality constraints.
        if C.size > 0:
            Aeq = C.copy()
            beq = d.copy()
        else:
            Aeq = np.empty((0, x.size))
            beq = np.empty((0,))
        if len(W) > 0:
            Aeq = np.vstack((Aeq, A[W, :]))
            beq = np.concatenate((beq, b[W]))
        
        # Solve the equality-constrained QP subproblem:
        #   min_p 0.5*pᵀGp + pᵀ*(Gx+c)   subject to   A_eq p = 0.
        if Aeq.shape[0] > 0:
            p, lambda_W = solve_eqp(G, G @ x + c, Aeq, np.zeros(Aeq.shape[0]))
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
            # Separate multipliers: first for equality constraints (from C) then for active inequalities.
            lambda_ineq = lambda_W[p_eq:] #if lambda_W.size > p_eq else np.array([])
            if all(lambda_ineq >= -tol):
                history.append((W.copy(), f_val, None, p_norm**2, x.copy()))
                lambda_dict = {i: lambda_ineq[j] for j, i in enumerate(W)}
                return x, lambda_dict, history
            else:
                # Remove the constraint with the most negative multiplier.
                lambda_ineq = np.array(lambda_ineq)
                j_index = np.argmin(lambda_ineq)
                j = W[j_index]
                print(f"Iteration {k}: p≈0 but lambda[{j}] = {lambda_ineq[j_index]:.4f} < 0; removing constraint {j}.")
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
                # For an inequality A[i]x ≥ b[i], a decrease in A[i]x is harmful.
                # If a_i @ p < -tol, then compute the maximum step to maintain feasibility.
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
