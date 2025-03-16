import numpy as np
import matplotlib as plt


# funkcja kwadratowa
# ax^2+bx+c
# d/dx 2ax+b

def f_kwadratowa(x):
    return np.sum(x**2)


def f_rossenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 -x[:-1])**2)

def f_ackley(x):
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / len(x))) - \
           np.exp(np.sum(np.cos(2 * np.pi * x)) / len(x)) + 20 + np.e


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