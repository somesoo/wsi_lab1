import pandas as pd
import numpy as np
import matplotlib as plt


# funkcja kwadratowa
# ax^2+bx+c
# d/dx 2ax+b

# funkcja kosztu -> suma kwadratów różnicy pomiędzy właściwą a przewidywaną wartością


def cost_function(true, predicted):
    cost = np.sum((true - predicted) ** 2) / len(true)
    return cost
