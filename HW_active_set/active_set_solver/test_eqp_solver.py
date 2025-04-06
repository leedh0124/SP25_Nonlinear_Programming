import numpy as np
from eqp_solver import solve_eqp

def test_eqp():
    """
    Test the solver on the quadratic program:
    
      minimize    x₁² + x₂²
      subject to  x₁ + x₂ = 5
    
    The problem is cast into the standard form:
         0.5 * xᵀ G x + cᵀ x
    by setting G = 2I (so that 0.5*(2x₁² + 2x₂²) = x₁² + x₂²) and c = 0.
    """
    # Define the problem data.
    G = 2 * np.eye(2)        # Quadratic term matrix.
    c = np.zeros(2)          # Zero linear term.
    A = np.array([[1, 1]])   # Equality constraint: x₁ + x₂ = 5.
    b = np.array([5])
    
    # Solve the KKT system.
    x, lam = solve_eqp(G, c, A, b)
    
    print("Test Problem: min x₁² + x₂² subject to x₁ + x₂ = 5")
    print("Solution x =", x)
    print("Lagrange multiplier λ =", lam)

if __name__ == "__main__":
    test_eqp()
