import json
import os
import math
import numpy as np
import cvxpy as cp
from geopy import distance

from solver_utils import create_index_matrix, generate_random_points, get_distance_matrix
from utils import evaluate_paths, print_result

################################################
# Задание ограничений
################################################
def get_constraints(X, u, drones_n, destinations_n, coords_n, opt={}):
    debug = opt.get('debug', False)

    ones = np.ones((destinations_n, 1))
    c = []
    arrival_i = 0
    departure_i = 0
    index_matrix = create_index_matrix(coords_n)

    if debug:
        print('Debugging constraints on index matrix:\n', index_matrix)

    # (1) Из строки departure_i должно выйти drones_n дронов
    # Сумма по строке отправления равна количеству дронов
    c += [X[departure_i,:] @ ones == drones_n]
    if debug:
        print('(1) Sum of row departure_i == drones_n:\n',
              index_matrix[departure_i,:])
    
    # (2) В точку возвращения должно вернуться drones_n дронов
    # Сумма по столбцу прибытия равна количеству дронов
    c += [X[:,arrival_i] @ ones == drones_n]
    if debug:
        print('(1) Sum of row arrival_i == drones_n:\n',
              index_matrix[:,arrival_i])

    # (3) В каждую точку (кроме 0) будет только один вход и только один выход
    c += [X[1:,:] @ ones == 1]
    c += [X[:,1:].T @ ones == 1]
    if debug:
        print(f'(3) Each string only contains one 1:\n',
              index_matrix[1:,:])
        print(f'(3) Each string only contains one 1:\n',
              index_matrix[:,1:].T)

    # (?)
    if departure_i != arrival_i:
        c += [X[departure_i, arrival_i] == 0]
        if debug:
            print(f'(?) dunno but sum == 0:\n',
                index_matrix[departure_i, arrival_i])

    # (4) Нет пути из точки в саму себя
    c += [cp.diag(X) == 0]
    if debug:
        print('(4) Each cell == 0:\n', np.diag(index_matrix))

    # (7) Точка отправления посещаются первыми
    c += [u[departure_i] == 1]

    # (8) Каждая другая точка посещается один раз
    for point_index in range(coords_n):
        if point_index != departure_i:
            c += [u[point_index] >= 2]
            c += [u[point_index] <= coords_n]

    # (9) Порядок посещения точек
    for i in range(1, destinations_n):
        for j in range(1, destinations_n):
            if i != j:
                c += [ u[i] - u[j] + 1  <= (destinations_n - 1) * (1 - X[i, j]) ]

    return c

################################################
# Решение задачи целочисленного программирования
################################################
def solve(drones_n, distance_matrix, destinations_n, coords, opt={}):
    debug = opt.get('debug', False)
    departure_i = 0
    print(f'Решаем для {drones_n} дрон{"a" if drones_n == 1 else "ов"}')
    # Определения переменных                                                    https://www.cvxpy.org/tutorial/intro/index.html#vectors-and-matrices
    X = cp.Variable(distance_matrix.shape, boolean=True)                        # Булевая матрица, где 1 означает наличие маршрута между точками i и j
    u = cp.Variable(destinations_n, integer=True)
    # Определение целевой функции
    objective = cp.Minimize(cp.sum(cp.multiply(distance_matrix, X)))
    constraints = get_constraints(X, u, drones_n, destinations_n, len(coords), opt)
    # Решение задачи
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)
    if X.value is None:
        raise RuntimeError('Невозможно найти решение')
    # Преобразование решения в маршруты
    X_sol = np.argwhere(np.isclose(X.value, 1.0, atol=1e-4))
    if debug:
        print('[DEBUG] X:\n', X.value)
        print('[DEBUG] u:\n', u.value)
        print('[DEBUG] X_sol:\n', X_sol)

    # Вывод длины оптимального маршрута
    ruta = print_result(X_sol, coords, departure_i, distance_matrix)
    return ruta

def find_optimal_amount_of_drones(coords, opt={}):
    destinations_n = len(coords)
    distance_matrix = get_distance_matrix(coords)
    lowest_result = float('inf')
    corresponding_i = -1
    departure_i = 0
    arrival_i = 0
    max_cycles = destinations_n if departure_i == arrival_i else destinations_n-1
    rutas = []

    for i in range(1, max_cycles):
        ruta = solve(i, distance_matrix, destinations_n, coords, opt)
        rutas.append(ruta)
        max_distance = max(ruta.items(), key=lambda x: x[1]['distance'])[1]['distance']

        if max_distance < lowest_result:
            lowest_result = max_distance
            corresponding_i = i

    evaluate_paths(rutas)
    return lowest_result, corresponding_i

"""
Reads, validates, and returns structured input data from a JSON file.

Expected JSON structure:
{
    "coords": [[latitude, longitude], ...]
}
"""
# TODO: Add max drones number
def get_optimal_solver_input(filepath):
    def is_valid_coord(coord):
        return (
            isinstance(coord, list) and
            len(coord) == 2 and
            all(isinstance(c, (int, float)) for c in coord)
        )

    def is_valid_list_of_coords(lst):
        return isinstance(lst, list) and all(is_valid_coord(item) for item in lst)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")

    required_keys = ["coords"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key: {key}")

    if not is_valid_list_of_coords(data["coords"]):
        raise ValueError("Invalid format for 'visits'. Expected list of [latitude, longitude].")

    return data

def get_optimal_random_solver_input(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")

    required_keys = ["coords_n"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key: {key}")

    if not isinstance(data["coords_n"], int):
        raise ValueError("Invalid type for 'coords_n'. Expected integer.")

    return data

def run_optimal_solver(data_filepath, opt={}):
    debug = opt.get('debug', False)
    random = opt.get('random', False)
    coords = []
    if random:
        coords = generate_random_points(get_optimal_random_solver_input(data_filepath)["coords_n"])
    else:
        coords = get_optimal_solver_input(data_filepath)["coords"]
    if debug:
        print("[DEBUG] Coords", coords)
    lowest_result, corresponding_i = find_optimal_amount_of_drones(coords, {'debug': debug})
    print(f'\nOptimal drones amount: {corresponding_i} ({np.round(lowest_result, 2)})km.')

"""
Reads, validates, and returns structured input data from a JSON file.

Expected JSON structure:
{
    "coords": [[latitude, longitude], ...],
    "drones_n": number
}
"""
def get_single_solver_input(filepath):
    def is_valid_coord(coord):
        return (
            isinstance(coord, list) and
            len(coord) == 2 and
            all(isinstance(c, (int, float)) for c in coord)
        )

    def is_valid_list_of_coords(lst):
        return isinstance(lst, list) and all(is_valid_coord(item) for item in lst)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")

    required_keys = ["coords", "drones_n"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key: {key}")

    if not is_valid_list_of_coords(data["coords"]):
        raise ValueError("Invalid format for 'coords'. Expected list of [latitude, longitude].")

    if not isinstance(data["drones_n"], int):
        raise ValueError("Invalid type for 'drones_n'. Expected integer.")

    return data

def run_single_solver(data_filepath, opt={}):
    debug = opt.get('debug', False)
    data = get_single_solver_input(data_filepath)
    coords = data["coords"]
    if debug:
        print("[DEBUG] Coords", coords)
    n = len(coords)
    C = np.zeros((n,n))
    for i in range(0, n):
        for j in range(0, len(coords)):
            C[i,j] = distance.distance(coords[i], coords[j]).km
    ruta = solve(data["drones_n"], C, n, coords, opt)
    print("Max drone distance", max(ruta.items(), key=lambda x: x[1]['distance'])[1]['distance'])
