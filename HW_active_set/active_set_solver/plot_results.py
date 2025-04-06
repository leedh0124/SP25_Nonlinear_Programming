import numpy as np
import matplotlib.pyplot as plt

def plot_feasible_region_and_contours(G, c, A, b, x_range=(-1, 3), y_range=(-1, 3), levels=50):
    # Create a grid of points.
    x1 = np.linspace(x_range[0], x_range[1], 400)
    x2 = np.linspace(y_range[0], y_range[1], 400)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Evaluate the objective function on the grid.
    Z = 0.5 * (G[0,0]*X1**2 + 2*G[0,1]*X1*X2 + G[1,1]*X2**2) + c[0]*X1 + c[1]*X2

    fig, ax = plt.subplots(figsize=(8,6))
    
    # Plot contour lines of the objective function.
    contour = ax.contour(X1, X2, Z, levels=levels, cmap='viridis')
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Compute feasibility mask for A*x ≥ b.
    feas_mask = np.ones_like(X1, dtype=bool)
    for i in range(A.shape[0]):
        feas_mask = feas_mask & ((A[i,0]*X1 + A[i,1]*X2) >= (b[i] - 1e-6))
    
    # Shade feasible region.
    ax.contourf(X1, X2, feas_mask, levels=[0.5, 1], colors='lightblue', alpha=0.5)
    
    # Plot the boundaries for the inequality constraints.
    for i in range(A.shape[0]):
        # If the constraint is not vertical:
        if abs(A[i,1]) > 1e-6:
            x_vals = np.linspace(x_range[0], x_range[1], 200)
            y_vals = (b[i] - A[i,0]*x_vals) / A[i,1]
            ax.plot(x_vals, y_vals, 'r--', label=f"Ineq {i+1}" if i == 0 else None)
        else:
            x_val = b[i] / A[i,0]
            ax.axvline(x=x_val, color='r', linestyle='--', label=f"Ineq {i+1}" if i == 0 else None)
    
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title('Objective Contours, Feasible Region, and Active-Set Iterates')
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    
    return fig, ax

