import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from autograd import grad



def f_kwadratowa(x):
    return np.dot(x, x)  

def f_rosenbrock(x):
    x = np.asarray(x)
    return np.dot(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2, np.ones_like(x[:-1]))

def f_ackley(x):
    n = x.shape[0]
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.dot(x, x) / n))
    term2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
    return term1 + term2 + 20 + np.e



def wykresy_funkcji_test():
    # Zakres wartości dla wykresów 2D
    x_vals = np.linspace(-5, 5, 100)
    y_vals = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Obliczenie wartości funkcji dla siatki punktów
    Z_quad = np.array([f_kwadratowa(np.array([x, y])) for x, y in zip(X.ravel(), Y.ravel())])
    Z_quad = Z_quad.reshape(X.shape)
    
    Z_rosen = np.array([f_rosenbrock(np.array([x, y])) for x, y in zip(X.ravel(), Y.ravel())])
    Z_rosen = Z_rosen.reshape(X.shape)
    
    Z_ackley = np.array([f_ackley(np.array([x, y])) for x, y in zip(X.ravel(), Y.ravel())])
    Z_ackley = Z_ackley.reshape(X.shape)

    # Tworzenie wykresów konturowych
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Wykres konturowy dla funkcji kwadratowej
    ax = axes[0]
    cp = ax.contourf(X, Y, Z_quad, levels=50, cmap='viridis')
    fig.colorbar(cp, ax=ax)
    ax.set_title("f_kwadratowa")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Wykres konturowy dla funkcji Rosenbrocka
    ax = axes[1]
    cp = ax.contourf(X, Y, Z_rosen, levels=50, cmap='viridis')
    fig.colorbar(cp, ax=ax)
    ax.set_title("f_rosenbrock")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Wykres konturowy dla funkcji Ackleya
    ax = axes[2]
    cp = ax.contourf(X, Y, Z_ackley, levels=50, cmap='viridis')
    fig.colorbar(cp, ax=ax)
    ax.set_title("f_ackley")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.show()
    
# Funkcja licząca pochodną
wykresy_funkcji_test()
gradient_func = grad(f_kwadratowa)
x_test = np.array([2.0])
print(gradient_func(x_test))

# funkcja kosztu -> suma kwadratów różnicy pomiędzy właściwą a przewidywaną wartością

def cost_function(true, predicted):
    cost = np.sum((true - predicted) ** 2) / len(true)
    return cost

# def solver(
    # eval_func: Callable[[Sequence[float]], float],
    # x0: Sequence[float],
    # params: SolverParameters,
    #...
# ) -> solver_result:
    # x_current
    # iteration
    # max_iteration

def gradient_prosty(stop_condition_1, stop_condition_2, iterations, ):
    
    
    
    print("func")