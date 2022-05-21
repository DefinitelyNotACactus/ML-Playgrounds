import numpy as np
from numpy.random import uniform

def make_checkers(grid_size=3, n_samples=100):
    X = np.zeros((n_samples, 2))
    y = np.zeros(n_samples)
    
    f_x_checkers = lambda yn : 1 if yn is True else 0
    
    current_y = False
    i = 0
    for x0_padding in range(grid_size):
        for x1_padding in range(grid_size):
            current_y = not current_y
            for point in range(n_samples//grid_size**2):
                X[i][0] = uniform(x0_padding, x0_padding + 1)
                X[i][1] = uniform(x1_padding, x1_padding + 1)
                y[i] = f_x_checkers(current_y)
                i += 1
    # Lidar com as sobras, adicionando ao último quadrado
    while i < n_samples:
        X[i][0] = uniform(x0_padding, x0_padding + 1)
        X[i][1] = uniform(x1_padding, x1_padding + 1)
        y[i] = f_x_checkers(current_y)
        i += 1
        
    return X, y
