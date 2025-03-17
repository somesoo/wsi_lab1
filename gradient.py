import autograd.numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from autograd import grad
import time
from dataclasses import dataclass
from typing import Callable, Sequence, List



def f_square(x: np.ndarray) -> float:
    return np.sum(x**2)

def f_rosenbrock(x: np.ndarray) -> float:
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def f_ackley(x: np.ndarray) -> float:
    n = x.shape[0]
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.exp(1)




def wykresy_funkcji_test():
    # Zakres wartości dla wykresów 2D
    x_vals = np.linspace(-5, 5, 100)
    y_vals = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Obliczenie wartości funkcji dla siatki punktów
    Z_quad = np.array([f_square(np.array([x, y])) for x, y in zip(X.ravel(), Y.ravel())])
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
    ax.set_title("f_square")
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

@dataclass
class SolverParameters:
    alpha: float
    max_iter: int
    tol: float

@dataclass
class SolverResult:
    x_opt: np.ndarray
    f_opt: float
    iterations: int
    success: bool
    history: List[float]

def solver(
    eval_func: Callable[[np.ndarray], float],
    x0: Sequence[float],
    params: SolverParameters
) -> SolverResult:
    
    x = np.array(x0, dtype=float)
    grad_func = grad(eval_func)
    history = []
    success = False

    start_time = time.time()
    
    for i in range(params.max_iter):
        f_val = eval_func(x)
        history.append(f_val)

        # Gradient
        g = grad_func(x)

        # Krok w przeciwną stronę do gradientu, z *przycięciem* do [-10,10]
        x_next = x - params.alpha * g
        x_next = np.clip(x_next, -10, 10)  # żeby nie "uciekać" w nieskończoność

        # Jeśli różnica kolejnych punktów jest niewielka → stop
        if np.linalg.norm(x_next - x) < params.tol:
            success = True
            x = x_next
            break

        x = x_next
        # warunek przerwania gdy wyniki są bez sensu
        if not np.isfinite(f_val):
            break
    
    end_time = time.time()
    total_time = end_time - start_time

    f_opt = eval_func(x)
    iterations = i + 1

    print(f"Solver zakończony w {iterations} iteracjach (czas: {total_time:.4f}s). "
          f"f_opt={f_opt:.4e}, ||x||={np.linalg.norm(x):.3f}, success={success}")
    
    return SolverResult(
        x_opt=x,
        f_opt=f_opt,
        iterations=iterations,
        success=success,
        history=history
    )

if __name__ == "__main__":

    # Parametry ogólne
    n = 10
    max_iter = 20000
    tol = 1e-6

    np.random.seed(65)
    x0 = np.random.uniform(-10, 10, size=n)

    test_funcs = [
        ("Quadratic", f_square),
        ("Rosenbrock", f_rosenbrock),
        ("Ackley", f_ackley),
    ]

    alphas = [1e-3, 1e-4, 1e-5]

    # results[func_name][alpha] = SolverResult
    results = {}

    # 1) Uruchamiamy solver dla każdej funkcji i każdej wartości alpha
    for func_name, func in test_funcs:
        results[func_name] = {}
        print(f"\n=== Test funkcji {func_name} ===")

        for alpha in alphas:
            print(f"--- alpha={alpha} ---")
            params = SolverParameters(alpha=alpha, max_iter=max_iter, tol=tol)
            result = solver(func, x0, params)

            results[func_name][alpha] = result  # Zachowujemy do późniejszego wykresu

            print(f"  Ostateczne f_opt = {result.f_opt:.6f}")
            print(f"  Ostateczne x_opt = {result.x_opt}")
            print(f"  Iterations = {result.iterations}, Success = {result.success}")

    for func_name in results.keys():
        plt.figure()
        plt.title(f"Przebieg wartości {func_name}")
        plt.xlabel("Iteracja")
        plt.ylabel("Wartość funkcji")

        # Rysujemy krzywą dla każdej alpha
        for alpha in alphas:
            solver_res = results[func_name][alpha]
            plt.plot(range(solver_res.iterations), solver_res.history, label=f"alpha={alpha}")

        plt.legend()
        plt.show()
