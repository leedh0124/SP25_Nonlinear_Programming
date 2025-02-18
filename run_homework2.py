import time
import numpy as np
import pandas as pd

from steepest_descent_method import steepest_descent
from Newton_method import Newton_method, Newton_modified_Cholesky_Added_Multiple_Identity

# Define optimization parameters
max_iter = 5000
tol = 1e-8
alp_init = 1
c_1 = 1e-4
tau = 0.5
beta = 1e-4

# Define the Rosenbrock function and its gradient and Hessian
def rosenbrock(x, n):  
    """
    Rosenbrock function
    :param x: 
        input vector
    :param n: 
        dimension of the problem
    :return: 
        value of the Rosenbrock function
    """
    return sum(100 * (x[i] - x[i-1]**2)**2 + (1 - x[i-1])**2 for i in range(1, n))

def rosenbrock_gradient(x, n):
    """
    Gradient of the Rosenbrock function
    :param x: 
        input vector
    :param n: 
        dimension of the problem
    :return: 
        gradient of the Rosenbrock function
    """
    grad_i = np.zeros(n)
    grad_ip1 = np.zeros(n)
    for i in range(0, n-1):
        grad_i[i] = -400 * x[i+1] * x[i] + 400 * x[i]**3 + 2 * (x[i] - 1)
        grad_ip1[i+1] = 200 * x[i+1] - 200 * x[i]**2
    grad = grad_i + grad_ip1
    return grad

def rosenbrock_hessian(x, n):
    """
    Hessian of the Rosenbrock function
    :param x: 
        input vector
    :param n: 
        dimension of the problem
    :return: 
        Hessian matrix of the Rosenbrock function
    """
    hess_i = np.zeros((n, n))
    hess_ip1 = np.zeros((n, n))
    for i in range(0, n-1):
        hess_i[i, i] = -400 * x[i+1] + 1200 * x[i]**2 + 2
        hess_i[i, i+1] = -400 * x[i]
        hess_ip1[i+1, i] = -400 * x[i]
        hess_ip1[i+1, i+1] = 200 
    hess = hess_i + hess_ip1
    return hess

results = pd.DataFrame(columns=['Question', 'Method', 'x_min', 'f_min', 'Number of iterations', 'Elapsed Time', 'Termination', "Relevance of Newton's method modification", "Local rate of convergence"])

# Question 1:
n = 2
x0 = np.array([-1.2, 1])

print("Question 1: Rosenbrock function with n=2")

print("\n Steepest Descent Method")

start_time = time.perf_counter()
x_min, f_min, n_iter = steepest_descent(rosenbrock, rosenbrock_gradient, x0, alp_init, c_1, tau, tol, max_iter, n)
elapsed_time = time.perf_counter() - start_time

if n_iter == max_iter:
    termination = "Maximum number of iterations reached"
else:
    termination = "Converged"

new_row = {
    'Question': 'Question 1',
    'Method': 'Steepest Descent',
    'x_min': x_min,
    'f_min': round(f_min, 6),
    'Number of iterations': n_iter,
    'Elapsed Time': elapsed_time,
    'Termination': termination,
    "Relevance of Newton's method modification": "NA",
    "Local rate of convergence": None
}
results.loc[len(results)] = new_row

print("\n Newton's Method")

start_time = time.perf_counter()
x_min, f_min, n_iter = Newton_method(rosenbrock, rosenbrock_gradient, rosenbrock_hessian, x0, alp_init, c_1, tau, tol, max_iter, n)
elapsed_time = time.perf_counter() - start_time

if n_iter == max_iter:
    termination = "Maximum number of iterations reached"
else:
    termination = "Converged"

new_row = {
    'Question': 'Question 1',
    'Method': "Newton's Method",
    'x_min': x_min,
    'f_min': round(f_min, 6),
    'Number of iterations': n_iter,
    'Elapsed Time': elapsed_time,
    'Termination': termination,
    "Relevance of Newton's method modification": "NA",
    "Local rate of convergence": None
}
results.loc[len(results)] = new_row

print("\n Modified Newton's Method using Cholesky with Added Multiple of the Identity algorithm")

start_time = time.perf_counter()
x_min, f_min, n_iter = Newton_modified_Cholesky_Added_Multiple_Identity(rosenbrock, rosenbrock_gradient, rosenbrock_hessian, x0, alp_init, c_1, tau, beta, tol, max_iter, n)
elapsed_time = time.perf_counter() - start_time

if n_iter == max_iter:
    termination = "Maximum number of iterations reached"
else:
    termination = "Converged"

cnew_row = {
    'Question': 'Question 1',
    'Method': "Modified Newton's Method with Cholesky",
    'x_min': x_min,
    'f_min': round(f_min, 6),
    'Number of iterations': n_iter,
    'Elapsed Time': elapsed_time,
    'Termination': termination,
    "Relevance of Newton's method modification": "NA",
    "Local rate of convergence": None
}
results.loc[len(results)] = new_row


# Question 2:
n = 10
x0 = np.ones(n) * -1

print("Question 2: Rosenbrock function with n=10")

print("\n Steepest Descent Method")

start_time = time.perf_counter()
x_min, f_min, n_iter = steepest_descent(rosenbrock, rosenbrock_gradient, x0, alp_init, c_1, tau, tol, max_iter, n)
elapsed_time = time.perf_counter() - start_time

if n_iter == max_iter:
    termination = "Maximum number of iterations reached"
else:
    termination = "Converged"

new_row = {
    'Question': 'Question 2',
    'Method': 'Steepest Descent',
    'x_min': x_min,
    'f_min': round(f_min, 6),
    'Number of iterations': n_iter,
    'Elapsed Time': elapsed_time,
    'Termination': termination,
    "Relevance of Newton's method modification": "NA",
    "Local rate of convergence": None
}
results.loc[len(results)] = new_row

print("\n Newton's Method")

start_time = time.perf_counter()
x_min, f_min, n_iter = Newton_method(rosenbrock, rosenbrock_gradient, rosenbrock_hessian, x0, alp_init, c_1, tau, tol, max_iter, n)
elapsed_time = time.perf_counter() - start_time

if n_iter == max_iter:
    termination = "Maximum number of iterations reached"
else:
    termination = "Converged"

new_row = {
    'Question': 'Question 2',
    'Method': "Newton's Method",
    'x_min': x_min,
    'f_min': round(f_min, 6),
    'Number of iterations': n_iter,
    'Elapsed Time': elapsed_time,
    'Termination': termination,
    "Relevance of Newton's method modification": "NA",
    "Local rate of convergence": None
}
results.loc[len(results)] = new_row

print("\n Modified Newton's Method using Cholesky with Added Multiple of the Identity algorithm")

start_time = time.perf_counter()
x_min, f_min, n_iter = Newton_modified_Cholesky_Added_Multiple_Identity(rosenbrock, rosenbrock_gradient, rosenbrock_hessian, x0, alp_init, c_1, tau, beta, tol, max_iter, n)
elapsed_time = time.perf_counter() - start_time

if n_iter == max_iter:
    termination = "Maximum number of iterations reached"
else:
    termination = "Converged"

new_row = {
    'Question': 'Question 2',
    'Method': "Modified Newton's Method with Cholesky",
    'x_min': x_min,
    'f_min': round(f_min, 6),
    'Number of iterations': n_iter,
    'Elapsed Time': elapsed_time,
    'Termination': termination,
    "Relevance of Newton's method modification": "NA",
    "Local rate of convergence": None
}
results.loc[len(results)] = new_row


# Question 3:
n = 10
x0 = np.ones(n) * 2

print("Question 3: Rosenbrock function with n=10")



print("\n Steepest Descent Method")

start_time = time.perf_counter()
x_min, f_min, n_iter = steepest_descent(rosenbrock, rosenbrock_gradient, x0, alp_init, c_1, tau, tol, max_iter, n)
elapsed_time = time.perf_counter() - start_time

if n_iter == max_iter:
    termination = "Maximum number of iterations reached"
else:
    termination = "Converged"

new_row = {
    'Question': 'Question 3',
    'Method': 'Steepest Descent',
    'x_min': x_min,
    'f_min': round(f_min, 6),
    'Number of iterations': n_iter,
    'Elapsed Time': elapsed_time,
    'Termination': termination,
    "Relevance of Newton's method modification": "NA",
    "Local rate of convergence": None
}
results.loc[len(results)] = new_row

print("\n Newton's Method")

start_time = time.perf_counter()
x_min, f_min, n_iter = Newton_method(rosenbrock, rosenbrock_gradient, rosenbrock_hessian, x0, alp_init, c_1, tau, tol, max_iter, n)
elapsed_time = time.perf_counter() - start_time

if n_iter == max_iter:
    termination = "Maximum number of iterations reached"
else:
    termination = "Converged"

new_row = {
    'Question': 'Question 3',
    'Method': "Newton's Method",
    'x_min': x_min,
    'f_min': round(f_min, 6),
    'Number of iterations': n_iter,
    'Elapsed Time': elapsed_time,
    'Termination': termination,
    "Relevance of Newton's method modification": "NA",
    "Local rate of convergence": None
}
results.loc[len(results)] = new_row

print("\n Modified Newton's Method using Cholesky with Added Multiple of the Identity algorithm")

start_time = time.perf_counter()
x_min, f_min, n_iter = Newton_modified_Cholesky_Added_Multiple_Identity(rosenbrock, rosenbrock_gradient, rosenbrock_hessian, x0, alp_init, c_1, tau, beta, tol, max_iter, n)
elapsed_time = time.perf_counter() - start_time

if n_iter == max_iter:
    termination = "Maximum number of iterations reached"
else:
    termination = "Converged"

new_row = {
    'Question': 'Question 3',
    'Method': "Modified Newton's Method with Cholesky",
    'x_min': x_min,
    'f_min': round(f_min, 6),
    'Number of iterations': n_iter,
    'Elapsed Time': elapsed_time,
    'Termination': termination,
    "Relevance of Newton's method modification": "NA",
    "Local rate of convergence": None
}
results.loc[len(results)] = new_row


# Question 4:
n = 100
x0 = np.ones(n) * -1

print("Question 4: Rosenbrock function with n=100")



print("\n Steepest Descent Method")

start_time = time.perf_counter()
x_min, f_min, n_iter = steepest_descent(rosenbrock, rosenbrock_gradient, x0, alp_init, c_1, tau, tol, max_iter, n)
elapsed_time = time.perf_counter() - start_time

if n_iter == max_iter:
    termination = "Maximum number of iterations reached"
else:
    termination = "Converged"

new_row = {
    'Question': 'Question 4',
    'Method': 'Steepest Descent',
    'x_min': x_min,
    'f_min': round(f_min, 6),
    'Number of iterations': n_iter,
    'Elapsed Time': elapsed_time,
    'Termination': termination,
    "Relevance of Newton's method modification": "NA",
    "Local rate of convergence": None
}
results.loc[len(results)] = new_row

print("\n Newton's Method")

start_time = time.perf_counter()
x_min, f_min, n_iter = Newton_method(rosenbrock, rosenbrock_gradient, rosenbrock_hessian, x0, alp_init, c_1, tau, tol, max_iter, n)
elapsed_time = time.perf_counter() - start_time

if n_iter == max_iter:
    termination = "Maximum number of iterations reached"
else:
    termination = "Converged"

new_row = {
    'Question': 'Question 4',
    'Method': "Newton's Method",
    'x_min': x_min,
    'f_min': round(f_min, 6),
    'Number of iterations': n_iter,
    'Elapsed Time': elapsed_time,
    'Termination': termination,
    "Relevance of Newton's method modification": "NA",
    "Local rate of convergence": None
}
results.loc[len(results)] = new_row

print("\n Modified Newton's Method using Cholesky with Added Multiple of the Identity algorithm")

start_time = time.perf_counter()
x_min, f_min, n_iter = Newton_modified_Cholesky_Added_Multiple_Identity(rosenbrock, rosenbrock_gradient, rosenbrock_hessian, x0, alp_init, c_1, tau, beta, tol, max_iter, n)
elapsed_time = time.perf_counter() - start_time

if n_iter == max_iter:
    termination = "Maximum number of iterations reached"
else:
    termination = "Converged"

new_row = {
    'Question': 'Question 4',
    'Method': "Modified Newton's Method with Cholesky",
    'x_min': x_min,
    'f_min': round(f_min, 6),
    'Number of iterations': n_iter,
    'Elapsed Time': elapsed_time,
    'Termination': termination,
    "Relevance of Newton's method modification": "NA",
    "Local rate of convergence": None
}
results.loc[len(results)] = new_row


# Question 5:
n = 100
x0 = np.ones(n) * 2

print("Question 5: Rosenbrock function with n=100")



print("\n Steepest Descent Method")

start_time = time.perf_counter()
x_min, f_min, n_iter = steepest_descent(rosenbrock, rosenbrock_gradient, x0, alp_init, c_1, tau, tol, max_iter, n)
elapsed_time = time.perf_counter() - start_time

if n_iter == max_iter:
    termination = "Maximum number of iterations reached"
else:
    termination = "Converged"

new_row = {
    'Question': 'Question 5',
    'Method': 'Steepest Descent',
    'x_min': x_min,
    'f_min': round(f_min, 6),
    'Number of iterations': n_iter,
    'Elapsed Time': elapsed_time,
    'Termination': termination,
    "Relevance of Newton's method modification": "NA",
    "Local rate of convergence": None
}
results.loc[len(results)] = new_row

print("\n Newton's Method")

start_time = time.perf_counter()
x_min, f_min, n_iter = Newton_method(rosenbrock, rosenbrock_gradient, rosenbrock_hessian, x0, alp_init, c_1, tau, tol, max_iter, n)
elapsed_time = time.perf_counter() - start_time

if n_iter == max_iter:
    termination = "Maximum number of iterations reached"
else:
    termination = "Converged"

new_row = {'Question': 'Question 5',
    'Method': "Newton's Method",
    'x_min': x_min,
    'f_min': round(f_min, 6),
    'Number of iterations': n_iter,
    'Elapsed Time': elapsed_time,
    'Termination': termination,
    "Relevance of Newton's method modification": "NA",
    "Local rate of convergence": None
}
results.loc[len(results)] = new_row

print("\n Modified Newton's Method using Cholesky with Added Multiple of the Identity algorithm")

start_time = time.perf_counter()
x_min, f_min, n_iter = Newton_modified_Cholesky_Added_Multiple_Identity(rosenbrock, rosenbrock_gradient, rosenbrock_hessian, x0, alp_init, c_1, tau, beta, tol, max_iter, n)
elapsed_time = time.perf_counter() - start_time

if n_iter == max_iter:
    termination = "Maximum number of iterations reached"
else:
    termination = "Converged"

new_row = {
    'Question': 'Question 5',
    'Method': "Modified Newton's Method with Cholesky",
    'x_min': x_min,
    'f_min': round(f_min, 6),
    'Number of iterations': n_iter,
    'Elapsed Time': elapsed_time,
    'Termination': termination,
    "Relevance of Newton's method modification": "NA",
    "Local rate of convergence": None
}
results.loc[len(results)] = new_row


# Question 6:
n = 1000
x0 = np.ones(n) * 2

print("Question 6: Rosenbrock function with n=1000")



print("\n Steepest Descent Method")

start_time = time.perf_counter()
x_min, f_min, n_iter = steepest_descent(rosenbrock, rosenbrock_gradient, x0, alp_init, c_1, tau, tol, max_iter, n)
elapsed_time = time.perf_counter() - start_time

if n_iter == max_iter:
    termination = "Maximum number of iterations reached"
else:
    termination = "Converged"

new_row = {
    'Question': 'Question 6',
    'Method': 'Steepest Descent',
    'x_min': x_min,
    'f_min': round(f_min, 6),
    'Number of iterations': n_iter,
    'Elapsed Time': elapsed_time,
    'Termination': termination,
    "Relevance of Newton's method modification": "NA",
    "Local rate of convergence": None
}
results.loc[len(results)] = new_row

print("\n Newton's Method")

start_time = time.perf_counter()
x_min, f_min, n_iter = Newton_method(rosenbrock, rosenbrock_gradient, rosenbrock_hessian, x0, alp_init, c_1, tau, tol, max_iter, n)
elapsed_time = time.perf_counter() - start_time

if n_iter == max_iter:
    termination = "Maximum number of iterations reached"
else:
    termination = "Converged"

new_row = {
    'Question': 'Question 6',
    'Method': "Newton's Method",
    'x_min': x_min,
    'f_min': round(f_min, 6),
    'Number of iterations': n_iter,
    'Elapsed Time': elapsed_time,
    'Termination': termination,
    "Relevance of Newton's method modification": "NA",
    "Local rate of convergence": None
}
results.loc[len(results)] = new_row

print("\n Modified Newton's Method using Cholesky with Added Multiple of the Identity algorithm")

start_time = time.perf_counter()
x_min, f_min, n_iter = Newton_modified_Cholesky_Added_Multiple_Identity(rosenbrock, rosenbrock_gradient, rosenbrock_hessian, x0, alp_init, c_1, tau, beta, tol, max_iter, n)
elapsed_time = time.perf_counter() - start_time

if n_iter == max_iter:
    termination = "Maximum number of iterations reached"
else:
    termination = "Converged"

new_row = {
    'Question': 'Question 6',
    'Method': "Modified Newton's Method with Cholesky",
    'x_min': x_min,
    'f_min': round(f_min, 6),
    'Number of iterations': n_iter,
    'Elapsed Time': elapsed_time,
    'Termination': termination,
    "Relevance of Newton's method modification": "NA",
    "Local rate of convergence": None
}
results.loc[len(results)] = new_row


# Question 7:
n = 1000
x0 = np.ones(n) * 2

print("Question 7: Rosenbrock function with n=1000")



print("\n Steepest Descent Method")

start_time = time.perf_counter()
x_min, f_min, n_iter = steepest_descent(rosenbrock, rosenbrock_gradient, x0, alp_init, c_1, tau, tol, max_iter, n)
elapsed_time = time.perf_counter() - start_time

if n_iter == max_iter:
    termination = "Maximum number of iterations reached"
else:
    termination = "Converged"

new_row = {
    'Question': 'Question 7',
    'Method': 'Steepest Descent',
    'x_min': x_min,
    'f_min': round(f_min, 6),
    'Number of iterations': n_iter,
    'Elapsed Time': elapsed_time,
    'Termination': termination,
    "Relevance of Newton's method modification": "NA",
    "Local rate of convergence": None
}
results.loc[len(results)] = new_row

print("\n Newton's Method")

start_time = time.perf_counter()
x_min, f_min, n_iter = Newton_method(rosenbrock, rosenbrock_gradient, rosenbrock_hessian, x0, alp_init, c_1, tau, tol, max_iter, n)
elapsed_time = time.perf_counter() - start_time

if n_iter == max_iter:
    termination = "Maximum number of iterations reached"
else:
    termination = "Converged"

new_row = {
    'Question': 'Question 7',
    'Method': "Newton's Method",
    'x_min': x_min,
    'f_min': round(f_min, 6),
    'Number of iterations': n_iter,
    'Elapsed Time': elapsed_time,
    'Termination': termination,
    "Relevance of Newton's method modification": "NA",
    "Local rate of convergence": None
}
results.loc[len(results)] = new_row

print("\n Modified Newton's Method using Cholesky with Added Multiple of the Identity algorithm")

start_time = time.perf_counter()
x_min, f_min, n_iter = Newton_modified_Cholesky_Added_Multiple_Identity(rosenbrock, rosenbrock_gradient, rosenbrock_hessian, x0, alp_init, c_1, tau, beta, tol, max_iter, n)
elapsed_time = time.perf_counter() - start_time

if n_iter == max_iter:
    termination = "Maximum number of iterations reached"
else:
    termination = "Converged"

new_row = {
    'Question': 'Question 7',
    'Method': "Modified Newton's Method with Cholesky",
    'x_min': x_min,
    'f_min': round(f_min, 6),
    'Number of iterations': n_iter,
    'Elapsed Time': elapsed_time,
    'Termination': termination,
    "Relevance of Newton's method modification": "NA",
    "Local rate of convergence": None
}
results.loc[len(results)] = new_row


# Beale function
def beale(x):
    """
    Beale function
    :param x: 
        input vector
    :return: 
        value of the Beale function
    """
    x1, x2 = x
    return (1.5 - x1 + x1 * x2) ** 2 + (2.25 - x1 + x1 * x2 ** 2) ** 2 + (2.625 - x1 + x1 * x2 ** 3) ** 2

def beale_gradient(x):
    """
    Gradient of the Beale function
    :param x: 
        input vector
    :return: 
        gradient of the Beale function
    """
    x1, x2 = x
    # Compute the individual components of the Beale function
    u1 = 1.5 - x1 + x1 * x2
    u2 = 2.25 - x1 + x1 * x2**2
    u3 = 2.625 - x1 + x1 * x2**3
    
    # Partial derivative with respect to x1
    grad_x1 = (2 * u1 * (-1 + x2) +
               2 * u2 * (-1 + x2**2) +
               2 * u3 * (-1 + x2**3))
    
    # Partial derivative with respect to x2
    grad_x2 = (2 * u1 * x1 +
               2 * u2 * (2 * x1 * x2) +
               2 * u3 * (3 * x1 * x2**2))
    
    return np.array([grad_x1, grad_x2])

def beale_hessian(x):
    """
    Hessian of the Beale function
    :param x: 
        input vector
    :return: 
        Hessian matrix of the Beale function
    """
    x1, x2 = x
    # Compute u1, u2, u3
    u1 = 1.5 - x1 + x1 * x2
    u2 = 2.25 - x1 + x1 * x2**2
    u3 = 2.625 - x1 + x1 * x2**3

    # First derivatives of u's
    u1_x1 = -1 + x2
    u1_x2 = x1

    u2_x1 = -1 + x2**2
    u2_x2 = 2 * x1 * x2

    u3_x1 = -1 + x2**3
    u3_x2 = 3 * x1 * x2**2

    # Second derivatives of u's
    # For u1
    u1_x1x2 = 1      # mixed derivative; u1_x1x1 and u1_x2x2 are 0.
    # For u2
    u2_x1x2 = 2 * x2
    u2_x2x2 = 2 * x1
    # For u3
    u3_x1x2 = 3 * x2**2
    u3_x2x2 = 6 * x1 * x2

    # Hessian entries:
    # f_x1x1: second derivative with respect to x1 twice.
    f_x1x1 = 2 * (u1_x1**2 + u2_x1**2 + u3_x1**2)
    
    # f_x1x2: mixed partial derivative.
    f_x1x2 = 2 * (u1_x1 * u1_x2 + u2_x1 * u2_x2 + u3_x1 * u3_x2) \
             + 2 * (u1 * u1_x1x2 + u2 * u2_x1x2 + u3 * u3_x1x2)
    
    # f_x2x2: second derivative with respect to x2 twice.
    f_x2x2 = 2 * (u1_x2**2 + u2_x2**2 + u3_x2**2) \
             + 2 * (0 + u2 * u2_x2x2 + u3 * u3_x2x2)  # u1_x2x2 is 0

    # Assemble the Hessian matrix (symmetric)
    H = np.array([[f_x1x1, f_x1x2],
                  [f_x1x2, f_x2x2]])
    return H


# Question 8:
x0 = np.array([1, 1])

print("Question 8: Beale function")



print("\n Steepest Descent Method")

start_time = time.perf_counter()
x_min, f_min, n_iter = steepest_descent(beale, beale_gradient, x0, alp_init, c_1, tau, tol, max_iter)
elapsed_time = time.perf_counter() - start_time

if n_iter == max_iter:
    termination = "Maximum number of iterations reached"
else:
    termination = "Converged"

new_row = {
    'Question': 'Question 8',
    'Method': 'Steepest Descent',
    'x_min': x_min,
    'f_min': round(f_min, 6),
    'Number of iterations': n_iter,
    'Elapsed Time': elapsed_time,
    'Termination': termination,
    "Relevance of Newton's method modification": "NA",
    "Local rate of convergence": None
}
results.loc[len(results)] = new_row

print("\n Newton's Method")

start_time = time.perf_counter()
x_min, f_min, n_iter = Newton_method(beale, beale_gradient, beale_hessian, x0, alp_init, c_1, tau, tol, max_iter)
elapsed_time = time.perf_counter() - start_time

if n_iter == max_iter:
    termination = "Maximum number of iterations reached"
else:
    termination = "Converged"

new_row = {
    'Question': 'Question 8',
    'Method': "Newton's Method",
    'x_min': x_min,
    'f_min': round(f_min, 6),
    'Number of iterations': n_iter,
    'Elapsed Time': elapsed_time,
    'Termination': termination,
    "Relevance of Newton's method modification": "NA",
    "Local rate of convergence": None
}
results.loc[len(results)] = new_row

print("\n Modified Newton's Method using Cholesky with Added Multiple of the Identity algorithm")

start_time = time.perf_counter()
x_min, f_min, n_iter = Newton_modified_Cholesky_Added_Multiple_Identity(beale, beale_gradient, beale_hessian, x0, alp_init, c_1, tau, beta, tol, max_iter)
elapsed_time = time.perf_counter() - start_time

if n_iter == max_iter:
    termination = "Maximum number of iterations reached"
else:
    termination = "Converged"

new_row = {
    'Question': 'Question 8',
    'Method': "Modified Newton's Method with Cholesky",
    'x_min': x_min,
    'f_min': round(f_min, 6),
    'Number of iterations': n_iter,
    'Elapsed Time': elapsed_time,
    'Termination': termination,
    "Relevance of Newton's method modification": "NA",
    "Local rate of convergence": None
}
results.loc[len(results)] = new_row


# Question 9:
x0 = np.array([0, 0])

print("Question 9: Beale function")



print("\n Steepest Descent Method")

start_time = time.perf_counter()
x_min, f_min, n_iter = steepest_descent(beale, beale_gradient, x0, alp_init, c_1, tau, tol, max_iter)
elapsed_time = time.perf_counter() - start_time

if n_iter == max_iter:
    termination = "Maximum number of iterations reached"
else:
    termination = "Converged"

new_row = {
    'Question': 'Question 9',
    'Method': 'Steepest Descent',
    'x_min': x_min,
    'f_min': round(f_min, 6),
    'Number of iterations': n_iter,
    'Elapsed Time': elapsed_time,
    'Termination': termination,
    "Relevance of Newton's method modification": "NA",
    "Local rate of convergence": None
}
results.loc[len(results)] = new_row

print("\n Newton's Method")

start_time = time.perf_counter()
x_min, f_min, n_iter = Newton_method(beale, beale_gradient, beale_hessian, x0, alp_init, c_1, tau, tol, max_iter)
elapsed_time = time.perf_counter() - start_time

if n_iter == max_iter:
    termination = "Maximum number of iterations reached"
else:
    termination = "Converged"

new_row = {
    'Question': 'Question 9',
    'Method': "Newton's Method",
    'x_min': x_min,
    'f_min': round(f_min, 6),
    'Number of iterations': n_iter,
    'Elapsed Time': elapsed_time,
    'Termination': termination,
    "Relevance of Newton's method modification": "NA",
    "Local rate of convergence": None
}
results.loc[len(results)] = new_row

print("\n Modified Newton's Method using Cholesky with Added Multiple of the Identity algorithm")

start_time = time.perf_counter()
x_min, f_min, n_iter = Newton_modified_Cholesky_Added_Multiple_Identity(beale, beale_gradient, beale_hessian, x0, alp_init, c_1, tau, beta, tol, max_iter)
elapsed_time = time.perf_counter() - start_time

if n_iter == max_iter:
    termination = "Maximum number of iterations reached"
else:
    termination = "Converged"

new_row = {
    'Question': 'Question 9',
    'Method': "Modified Newton's Method with Cholesky",
    'x_min': x_min,
    'f_min': round(f_min, 6),
    'Number of iterations': n_iter,
    'Elapsed Time': elapsed_time,
    'Termination': termination,
    "Relevance of Newton's method modification": "NA",
    "Local rate of convergence": None
}
results.loc[len(results)] = new_row

def the_function(x, n):
    """
    The function to be optimized
    :param x: 
        input vector
    :param n: 
        dimension of the input vector
    :return: 
        value of the function
    """
    return x[0]**2 + sum((x[i]-x[i+1])**(2*(i+1)) for i in range(0, n-1))

def the_function_gradient(x, n):
    """
    Gradient of the function
    :param x: 
        input vector
    :param n:
        dimension of the input vector
    :return:
        gradient of the function
    """
    grad = np.zeros(n)
    grad[0] = 2 * x[0]
    for i in range(0, n-1):
        grad[i] += 2 * (i+1) * (x[i] - x[i+1])**(2*(i+1) - 1)
        grad[i+1] -= 2 * (i+1) * (x[i] - x[i+1])**(2*(i+1) - 1)
    return grad

def the_function_hessian(x, n):
    """
    Hessian of the function
    :param x: 
        input vector
    :param n:
        dimension of the input vector
    :return:
        Hessian matrix of the function
    """
    H = np.zeros((n, n))
    H[0,0] = 2
    for i in range(0, n-1):
        H[i,i] += 2 * (i+1) * (2*(i+1) - 1) * (x[i] - x[i+1])**(2*(i+1) - 2)
        H[i,i+1] -= 2 * (i+1) * (2*(i+1) - 1) * (x[i] - x[i+1])**(2*(i+1) - 2)
        H[i+1,i] -= 2 * (i+1) * (2*(i+1) - 1) * (x[i] - x[i+1])**(2*(i+1) - 2)
        H[i+1,i+1] += 2 * (i+1) * (2*(i+1) - 1) * (x[i] - x[i+1])**(2*(i+1) - 2)
    return H

# Question 10:
n = 10
x0 = np.arange(1, n+1)

print("Question 10: Beale function")



print("\n Steepest Descent Method")

start_time = time.perf_counter()
x_min, f_min, n_iter = steepest_descent(the_function, the_function_gradient, x0, alp_init, c_1, tau, tol, max_iter, n)
elapsed_time = time.perf_counter() - start_time

if n_iter == max_iter:
    termination = "Maximum number of iterations reached"
else:
    termination = "Converged"

new_row = {
    'Question': 'Question 10',
    'Method': 'Steepest Descent',
    'x_min': x_min,
    'f_min': round(f_min, 6),
    'Number of iterations': n_iter,
    'Elapsed Time': elapsed_time,
    'Termination': termination,
    "Relevance of Newton's method modification": "NA",
    "Local rate of convergence": None
}
results.loc[len(results)] = new_row

print("\n Newton's Method")

start_time = time.perf_counter()
x_min, f_min, n_iter = Newton_method(the_function, the_function_gradient, the_function_hessian, x0, alp_init, c_1, tau, tol, max_iter, n)
elapsed_time = time.perf_counter() - start_time

if n_iter == max_iter:
    termination = "Maximum number of iterations reached"
else:
    termination = "Converged"

new_row = {
    'Question': 'Question 10',
    'Method': "Newton's Method",
    'x_min': x_min,
    'f_min': round(f_min, 6),
    'Number of iterations': n_iter,
    'Elapsed Time': elapsed_time,
    'Termination': termination,
    "Relevance of Newton's method modification": "NA",
    "Local rate of convergence": None
}
results.loc[len(results)] = new_row

print("\n Modified Newton's Method using Cholesky with Added Multiple of the Identity algorithm")

start_time = time.perf_counter()
x_min, f_min, n_iter = Newton_modified_Cholesky_Added_Multiple_Identity(the_function, the_function_gradient, the_function_hessian, x0, alp_init, c_1, tau, beta, tol, max_iter, n)
elapsed_time = time.perf_counter() - start_time

if n_iter == max_iter:
    termination = "Maximum number of iterations reached"
else:
    termination = "Converged"

new_row = {
    'Question': 'Question 10',
    'Method': "Modified Newton's Method with Cholesky",
    'x_min': x_min,
    'f_min': round(f_min, 6),
    'Number of iterations': n_iter,
    'Elapsed Time': elapsed_time,
    'Termination': termination,
    "Relevance of Newton's method modification": "NA",
    "Local rate of convergence": None
}
results.loc[len(results)] = new_row

results.to_csv('results.csv', index=False)