import numpy as np
import matplotlib.pyplot as plt

def penalized_F(x, mu):
    """
    1. g is the penalized function.
    2. x.dot(x) = x1.x1 + x2.x2 + x3.x3, 
       sice .dot calculates dot product with itself.
    """
    g = x[0] + 3*x[1] + 2*x[2] -12
    return x.dot(x) + mu * g**2 

def grad_F(x, mu):
    """Gradient F(X) or penalized function."""
    g = x[0] + 3*x[1] + 2*x[2] -12
    return 2*x + 2*mu*g*np.array([1, 3, 2])

def line_search(x, p, mu, grad, alpha0=1.0, c1=1e-4, reduct=0.5):
    """
    Backtracking Line Search (Armijo condition).

    x : Current point.
    p : Search direction (-H * grad), approximation to Newton step.
    mu : Penalty parameter in penalized function F(x).
    grad : Gradient of F(x) at point x.
    alpha0 : Initial step size (typically set to 1.0).
    c1 : Parameter controlling sufficient decrease (typically set to 1e-4).
    reduct : Reduction factor for alpha if condition fails (typically 0.5).

    Returns:
    alpha : Step size satisfying Armijo condition.
    """
    print(f"{'*' * 90}")
    print(f"Applying line search for (α) alpha...\n")
    fx = penalized_F(x, mu)
    alpha = alpha0

    while penalized_F(x + alpha*p, mu) > fx + c1*alpha*np.dot(grad, p):
        alpha *= reduct
        print(f"Armijo condition not satisfied!\n reducing(α) alpha...\n new alpha = {alpha}\n")
    print(f"{'*' * 90}")
    return alpha

def dfp(x0, mu=1000.0, tol=1e-6, max_iter=100):
    """
    init DFP: (1)stop criterion -> (2)direction search -> (3)line search 
              -> (4)update x, grad, s(step)and y(grads diff) -> (5)update H 
              -> (6)prepare next iteration.
    """
    print(f"{'-' * 90}")
    print(f"Applying DFP method\n")
    x = x0.astype(float)
    n = len(x)
    H = np.eye(n) # Identity matrix n x n dim.
    grad = grad_F(x, mu)
    print(f"x0 = {x}\ngrad F(x0) = {grad}")
    history_F = [penalized_F(x, mu)]

    for k in range(max_iter):
        print(f"{'=' * 50}")
        if np.linalg.norm(grad) < tol: #(1)
            print(f"Gradient norm to small stopping algorithm!\n")
            break
        
        p = -H.dot(grad) #(2)
        alpha = line_search(x, p, mu, grad) #(3)
        print(f"p{k} = {p}\nalpha {alpha}\n")

        #(4)
        x_new = x + alpha*p
        grad_new = grad_F(x_new, mu)
        s = x_new - x
        y = grad_new - grad
        print(f"x({k+1}) = {x_new}\ngrad({k+1})= {grad_new}\n")
        print(f"s({k+1}) = {s}\ny({k+1}) = {y}\n")

        #(5)
        Hy = H.dot(y)
        sy = s.dot(y)
        yHy = y.dot(Hy)

        if sy > 1e-12 and yHy > 1e-12: #avoid division by "0" or a number close to it.
            H = H + np.outer(s, s)/sy -np.outer(Hy, Hy)/yHy
        
        print(f"H({k+1}) = {H}\n")
        x, grad = x_new, grad_new #(6)

        history_F.append(penalized_F(x, mu))
        print(f"{'=' * 50}")
    
    print(f"DFP converged in: {k+1}")
    print(f"Approximate solution x = {x}")
    print(f"F(x) = {penalized_F(x, mu)}") 
    print(f"{'-' * 90}")
    
    plt.figure(figsize=(9, 6))
    plt.plot(history_F, marker='o')
    plt.title("Convergence of DFP")
    plt.xlabel("Iterations")
    plt.ylabel("F(x)")
    plt.grid(True)
    plt.show()

if __name__=="__main__":
    mu = 1000.0
    x0 = np.zeros(3)
    dfp(x0=x0, mu=mu)