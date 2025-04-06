import numpy as np
from active_set import active_set_qp
from plot_results import plot_feasible_region_and_contours
import matplotlib.pyplot as plt


def run_test(G, c, A, b, C, d, starting_point, idx):
    
    print("\n======================================")
    print(f"Testing with starting point: {starting_point}")
    x0 = np.array(starting_point)
    x_opt, lambda_opt, history = active_set_qp(G, c, A, b, C, d, x0)
    
    print("\nFinal solution:")
    print("x* =", x_opt)
    print("Optimal Lagrange multipliers (for active constraints):")
    print(lambda_opt)
    
    print("\nIteration history (Working set, f(x), α, ||p||²):")
    for k, (W, f_val, alpha, p_norm_sq, x_iterate) in enumerate(history):
        print(f"Iteration {k}: W = {W}, f(x) = {f_val:.4f}, α = {alpha}, ||p||² = {p_norm_sq:.4e}, x = {x_iterate}")
        
    # Plot the iterates.
    fig, ax = plot_feasible_region_and_contours(G, c, A, b, C, d)
    colors = ['magenta', 'orange', 'cyan']  # one color per starting point
    markers = ['o', 's', '^']
    
    # Get the iterates including the starting point.
    iterates = get_iterates(history, x0)
    
    # Plot the iterates as a line with markers.
    ax.plot(iterates[:,0], iterates[:,1],
            marker=markers[idx], color=colors[idx],
            linestyle='-', linewidth=2, markersize=8,
            label=f"Start {sp}")
    
    # Mark the final solution clearly.
    ax.plot(iterates[-1,0], iterates[-1,1],
            marker='*', color=colors[idx], markersize=15)
    
    ax.legend()
    plt.title(f"Iterates for starting point {starting_point}")
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.savefig(f"./HW_active_set/active_set_solver/Iterates for starting point{idx}.png")
    plt.show()
    
def get_iterates(history, x0):
    """
    Given the history (a list of tuples where the 5th element is the iterate x),
    return a NumPy array of the iterates, including the starting point.
    """
    iterates = [x0]
    for record in history:
        # record = (W, f_val, alpha, ||p||², x_iterate)
        iterates.append(record[4])
    return np.array(iterates)

if __name__ == "__main__":
    # (a) Define QP data.
    G = np.array([[2, -2],
                  [-2, 4]])
    c = np.array([-2, -6])
    A = np.array([
        [-0.5, -0.5],
        [1, -2],
        [1, 0],
        [0, 1]
    ])
    b = np.array([-1.0, -2.0, 0.0, 0.0])
    C = np.array([])
    d = np.array([])
    # Test with starting points (0.5,0.5), (0,0), and (1,0).
    for idx, sp in enumerate([(0.5, 0.5), (0, 0), (1, 0)]):
        run_test(G, c, A, b, C, d, sp, idx)
        
    # (b) Define QP data.
        # (a) Define QP data.
    G = np.array([[2, 0],
                  [0, 2]])
    c = np.array([-6, -4])
    A = np.array([
        [-1, -1],
        [1, 0],
        [0, 1]
    ])
    b = np.array([-3.0, 0.0, 0.0])
    C = np.array([])
    d = np.array([])
    # Test with starting points (0.5,0.5), (0,0), and (1,0).
    for idx, sp in enumerate([(0.5, 0.5), (0, 0), (1, 0)]):
        run_test(G, c, A, b, C, d, sp, idx)
        
    # (Misc) Define QP data.
    G = np.array([[2, 0],
                  [0, 4]])
    c = np.array([-1, -2])
    A = np.array([
        [1, 0],
        [0, -1],
        [-1, 3],
        [-1, -1]
    ])
    b = np.array([1, -3, -1, -5])
    C = np.array([[1, 1]])
    d = np.array([3])
    # Test with starting points (0.5,0.5), (0,0), and (1,0).
    for idx, sp in enumerate([(1, 2), (2.5, 0.5), (2, 1)]):
        run_test(G, c, A, b, C, d, sp, idx)
    
    