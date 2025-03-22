import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from geopy import distance

def get_distance_matrix(coords):
    n = len(coords)
    C = np.zeros((n,n))

    for i in range(0, n):
        for j in range(0, n):
            C[i,j] = distance.distance(coords[i], coords[j]).km
    return C

def get_vars_and_obj(distance_matrix):
    # https://www.cvxpy.org/tutorial/intro/index.html#vectors-and-matrices
    # Булевая матрица, где 1 означает наличие маршрута между точками i и j
    X = cp.Variable(distance_matrix.shape, boolean=True)
    # Определение целевой функции
    objective = cp.Minimize(cp.sum(cp.multiply(distance_matrix, X)))
    u = cp.Variable(len(distance_matrix), integer=True)
    return X, u, objective

def solve_problem(objective, constraints, X):
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)

    if X.value is None:
        raise RuntimeError('Невозможно найти решение')

    # Преобразование решения в маршруты
    return np.argwhere(X.value==1)

def create_index_matrix(n):
    return np.array([[f"{i},{j}" for j in range(n)] for i in range(n)])

# Функция для генерации случайного числа с 6 знаками после запятой в заданном диапазоне
def generate_random_number(min_val, max_val):
    return round(np.random.uniform(min_val, max_val), 6)

def generate_random_points(n):
    array = np.zeros((n, 2))

    for i in range(n):
        array[i][0] = generate_random_number(12.000000, 12.200000)
        array[i][1] = generate_random_number(77.200000, 76.800000)

    return array
