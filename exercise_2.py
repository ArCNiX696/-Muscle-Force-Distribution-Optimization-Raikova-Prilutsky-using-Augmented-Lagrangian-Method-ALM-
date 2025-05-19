import numpy as np
import matplotlib.pyplot as plt

"""
Paper data: 
array A len = 9.
array d len = 12 

{d3a = d[2], d3k = d[3], d5k = d[5], 
 d5h = d[6], d7k = d[8] , d7h = d[8] }
"""
A = np.array([11.5, 92.5, 44.3, 98.1, 20.1, 6.1, 45.5, 31.0, 44.3], dtype=float)
d = np.array([0.0298, 0.0440, 0.044, 0.0138, 0.0329, 0.0329, 0.0279, 0.0250, 0.0250, 0.0619, 0.0317, 0.0368], dtype=float)
M = np.array([7.5, 5, 2.5], dtype=float)

def obj_function(F):
    return np.sum((F / A)**2)

def constraints(F):
    g1 = d[0]*F[0] -d[1]*F[1] - d[2]*F[2] - M[0]
    g2 = -d[3]*F[2] + d[4]*F[3] + d[5]*F[4] - d[7]*F[5] - d[8]*F[6] - M[1]
    g3 = d[6]*F[4] - d[9]*F[6] + d[10]*F[7] - d[11]*F[8] - M[2]
    return np.array([g1, g2, g3])

#(ALM) Penalized objective function.
def pnl_function(F, lambdas, pnl):
    g = constraints(F)
    lagrange = np.dot(lambdas, g)
    sqr_pnl = (pnl/2) * np.sum(g**2)
    return obj_function(F) + lagrange + sqr_pnl

# (ALM) Gradient of penalized fuction.
def grad_pln_function(F, lambdas, pnl, grad_bound = 1e4):
    obj_grad = 2*F / (A**2) 
    g = constraints(F)

    # Jacobian matrix --> Row: constrant, col: Fi
    J = np.zeros((3, 9)) 
    # g(1) 
    J[0, 0] = d[0]
    J[0, 1] = -d[1]
    J[0, 2] = -d[2]
    # g(2) 
    J[1, 2] = -d[3]
    J[1, 3] =  d[4]
    J[1, 4] =  d[5]
    J[1, 5] = -d[7]
    J[1, 6] = -d[8]
    # g(3) 
    J[2, 4] =  d[6]
    J[2, 6] = -d[9]
    J[2, 7] =  d[10]
    J[2, 8] = -d[11]
    
    obj_grad += J.T @ (lambdas + pnl * g)
    obj_grad = np.clip(obj_grad, -grad_bound, grad_bound)
    return obj_grad

def pos_condition(F):
    """
    Ensure all Fi ≥ 0, as required by the muscle force model.
    If verbose is True, print any values adjusted.
    """
    return np.maximum(F, 0) 

def AML(i=9,
        init_guess = 3.0,
        lmds=3,
        pnl=1e2,
        max_outer =100,
        max_inner=500,
        lr=1e-4,
        tol=1e-6):
    print(f"{'='*80}")
    print(f"Starting ALM optimization...")

    # Declare Initial guess, lambdas, penalty.
    F = np.ones(i) * init_guess
    lambdas = np.zeros(lmds)
    penalty = pnl
    obj_history = []
    pnl_history = []
    g_history = []

    for out_i in range(max_outer):
        print(f"\nOuter Iteration {out_i+1}")

        for inner_i in range(max_inner):
            grad = grad_pln_function(F, lambdas, penalty)
            F -= lr * grad
            F = pos_condition(F)

            if np.linalg.norm(grad) < 1e-4:
                print(f"  Inner convergence reached at iteration {inner_i+1}")
                break

        obj_val = obj_function(F)       
        pnl_func_val = pnl_function(F, lambdas, penalty)
        g = constraints(F)
        constraint_violation = np.linalg.norm(g)
        g_history.append(constraint_violation)
        pnl_history.append(pnl_func_val)
        obj_history.append(obj_val)
        print(f"Objective Fuction value: {pnl_func_val:.4f}")
        print(f"Constraint violation norm: {constraint_violation:.2e}")

        if constraint_violation < tol:
            print("Convergence achieved based on constraint satisfaction!\n")
            break

        lambdas += penalty * g
        penalty *=2
        print(f"New lambdas = [{lambdas}]")
        print(f"New penalty vaule = {penalty}")
        print(f"\n{'-'*80}")

    print(f"{'*'*80}")
    print(f"Optimization report:")
    print(f"\nOptimization finished after {out_i+1} outer iterations.")
    print(f"\nOptimal muscle forces:\n")
    for i in range(len(F)):
        print(f"F{i+1} = {F[i]}")
    print(f"\nFinal objective value: {pnl_func_val:.4f}")
    print(f"\n{'*'*80}")
    print(f"\n{'='*80}")
        
    return F, obj_history, pnl_history, g_history
if __name__=="__main__":
    F_opt, obj_history, pnl_history, g_history = AML()

    iterations = np.arange(len(obj_history))

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].plot(iterations, obj_history, marker='o', color="blue", label="Z(F): original objective")
    axs[0].plot(iterations, pnl_history, marker='o', color="green", label="Lagrangian: penalized objective")
    axs[0].set_title("Objective vs Penalized Objective")
    axs[0].set_xlabel("Iterations")
    axs[0].set_ylabel("Objective Value")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(iterations, g_history, marker='o', color="red", label="∥g(F)∥: constraint violation")
    axs[1].set_title("Constraint Violation Over Iterations")
    axs[1].set_xlabel("Iterations")
    axs[1].set_ylabel("∥g(F)∥")
    axs[1].set_yscale("log")  # Log scale for clarity
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()





