import autograd.numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from autograd import grad
import time



def f_square(x: np.ndarray) -> float:
    return np.sum(x**2)

def f_rosenbrock(x: np.ndarray) -> float:
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def f_ackley(x: np.ndarray) -> float:
    n = x.shape[0]
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e




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


def solver(eval_func, x0, learning_rate=0.0001, iterations=10000, stop_condition_2=1e-6):
    # eval_func funkcja do liczenia
    # x0 punkt startowy
    # learning_rate krok uczenia
    # iterations maksymalna liczba iteracji
    # stop_condition_2 wartość która wystarczy jako bliska

    gradient_func = grad(eval_func)
    x = np.array(x0)
    trajectory = [x.copy()]

    for i in range(iterations):
        gradient_x = gradient_func(x)
        x = x - learning_rate * gradient_x
        x = np.clip(x, -10, 10)
        trajectory.append(x.copy())

        if np.linalg.norm(gradient_x) < stop_condition_2:
            break

    return x, np.array(trajectory)


n = 10
alphas = [0.001, 0.010, 0.100]
# ustawiony punkt startowy
# x0 = np.array([5.0, 2.0])
# losowy punkt startowy
x0 = np.random.uniform(-10, 10, size=n)
#x0 = np.clip(x0, -10, 10)

functions = {
    "Kwadratowa": (lambda x: f_square(x)),
    "Rosenbrock": (lambda x: f_rosenbrock(x)),
    "Ackley": (lambda x: f_ackley(x))
}

# Uruchamianie optymalizacji bez wizualizacji
results = {}
for name, func in functions.items():
    for alpha in alphas:
        x_min, trajectory = solver(func, x0, alpha)
        results[(name, alpha)] = x_min

# Wypisanie wyników optymalizacji
for (name, alpha), x_min in results.items():
    print(f'Funkcja: {name}, alpha={alpha}, Optimum: {x_min}')


# for name, func in functions.items():
#     plt.figure(figsize=(8, 5))
#     for alpha in alphas:
#         _, values, _ = solver(func, x0, learning_rate=alpha)
#         plt.plot(values, label=f'alpha={alpha}')
    
#     plt.title(f'Zbieżność - {name}')
#     plt.xlabel('Iteracja')
#     plt.ylabel('Wartość funkcji celu')
#     plt.yscale('log')
#     plt.legend()
#     plt.grid()
#     plt.show()

# # Wykres trajektorii dla n=2 jeśli chcemy wizualizację w 2D
# if n == 2:
#     func_name = "Kwadratowa"  # Można zmienić na inną funkcję
#     x_min, _, trajectory = solver(functions[func_name], x0, alpha=1)
    
#     # Siatka dla konturów
#     x_vals = np.linspace(-10, 10, 100)
#     y_vals = np.linspace(-10, 10, 100)
#     X, Y = np.meshgrid(x_vals, y_vals)
#     Z = np.array([[functions[func_name](np.array([x, y])) for x in x_vals] for y in y_vals])
    
#     plt.figure(figsize=(8, 6))
#     plt.contour(X, Y, Z, levels=50, cmap='viridis')
#     plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-o', label='Ścieżka optymalizacji')
#     plt.scatter(x0[0], x0[1], color='blue', marker='o', label='Start')
#     plt.scatter(x_min[0], x_min[1], color='red', marker='x', label='Minimum')
#     plt.title(f'Trajektoria optymalizacji - {func_name}')
#     plt.xlabel('x1')
#     plt.ylabel('x2')
#     plt.legend()
#     plt.grid()
#     plt.show()