import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# funkcja kwadratowa
# ax^2+bx+c
# d/dx 2ax+b


def f_kwadratowa(x):
    return np.sum(x**2, axis=-1)

def f_rosenbrock(x):
    x = np.asarray(x)
    return np.sum(100 * (x[..., 1:] - x[..., :-1]**2)**2 + (1 - x[..., :-1])**2, axis=-1)

def f_ackley(x):
    n = x.shape[-1]
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2, axis=-1) / n)) - \
           np.exp(np.sum(np.cos(2 * np.pi * x), axis=-1) / n) + 20 + np.e


def wykresy_funkcji_test():

    # Zakres wartości dla wykresów 2D
    x_vals = np.linspace(-5, 5, 100)
    y_vals = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Obliczenie wartości funkcji dla siatki punktów
    Z_quad = f_kwadratowa(np.stack([X, Y], axis=-1))
    Z_rosen = f_rosenbrock(np.stack([X, Y], axis=-1))
    Z_ackley = f_ackley(np.stack([X, Y], axis=-1))

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