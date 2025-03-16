import autograd.numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from autograd import grad
import time



def f_square(x):
    return np.dot(x, x)

def f_rosenbrock(x):
    x = np.asarray(x)
    return np.dot(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2, np.ones_like(x[:-1]))

def f_ackley(x):
    n = len(x)
    norm_x = np.sqrt(np.dot(x, x) / n)
    term1 = -20 * np.exp(-0.2 * norm_x)
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
# wykresy_funkcji_test()


def solver(eval_func, x0, learning_rate=0.0001, iterations=10000, stop_condition_2=1e-6):
    # eval_func funkcja do liczenia
    # x0 punkt startowy
    # learning_rate krok uczenia
    # iterations maksymalna liczba iteracji
    # stop_condition_2 wartość która wystarczy jako bliska

    gradient_func = grad(eval_func)
    iterations = iterations
    learning_rate = learning_rate
    x = np.array(x0, dtype=np.float64)

    previous = []

    for i in range(iterations):
        gradient_x = gradient_func(x)
        x = x - learning_rate * gradient_x
        previous.append(gradient_func(x))

        if i > 0 and abs(previous[-1] - previous[-2]) < stop_condition_2:
            break

    return x, previous


alphas = [1, 10, 100]
x0 = np.array([5.0, -3.0])

plt.figure(figsize=(10, 5))

for alpha in alphas:
    start_time = time.time()
    _, history = solver(f_square, x0=alpha)
    elapsed_time = time.time() - start_time
    plt.plot(history, label=f'α={alpha}, czas={elapsed_time:.4f}s')

plt.xlabel("Iteracje")
plt.ylabel("Wartość funkcji celu")
plt.title("Zbieżność solvera dla f")
plt.legend()
plt.show()