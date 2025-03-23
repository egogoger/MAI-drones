import json
import os
import numpy as np
import cvxpy as cp
from collections import defaultdict
from geopy import distance

from utils import print_result

def extract_routes(X_sol, departure_index, drones_n):

    # Build graph
    graph = defaultdict(list)
    for i, j in X_sol:
        graph[i].append(j)

    routes = []
    for _ in range(drones_n):
        route = [departure_index]
        current = departure_index
        while graph[current]:
            nxt = graph[current].pop()
            route.append(nxt)
            current = nxt
        routes.append(route)
    return routes

################################################
# Задание ограничений
################################################
def get_constraints(X, u, drones_n, destinations_n, coords_n, departure_index=0):
    ones = np.ones((destinations_n, 1))
    constraints = []
    arrival_index = 0

    # (1) Из строки departure_index должно выйти drones_n дронов
    # Сумма по строке отправления равна количеству дронов
    constraints += [X[departure_index,:] @ ones == drones_n]
    
    # (2) В точку возвращения должно вернуться drones_n дронов
    # Сумма по столбцу прибытия равна количеству дронов
    constraints += [X[:,arrival_index] @ ones == drones_n]

    # В каждую точку (кроме 0) будет только один вход и только один выход
    if departure_index == 0:
        constraints += [X[1:,:] @ ones == 1]
        constraints += [X[:,1:].T @ ones == 1]

    if departure_index == destinations_n-1:
        constraints += [X[1:departure_index,:] @ ones == 1]
        constraints += [X[:,1:departure_index].T @ ones == 1]

    if departure_index != arrival_index:
        constraints += [X[departure_index, arrival_index] == 0]

    constraints += [cp.diag(X) == 0]                  # Точка не связана сама с собой
    constraints += [u[departure_index] == 1]          # Точка отправления посещается первой

    for point_index in range(coords_n):
        if point_index != departure_index:
            constraints += [u[point_index] >= 2]      # Каждая другая точка посещается один раз
            constraints += [u[point_index] <= coords_n]

    for i in range(1, destinations_n):
        for j in range(1, destinations_n):
            if i != j:
                constraints += [ u[i] - u[j] + 1  <= (destinations_n - 1) * (1 - X[i, j]) ]
    return constraints

################################################
# Решение задачи целочисленного программирования
################################################
def solve(drones_n, distance_matrix, destinations_n, coords, opt={}):
    debug = opt.get('debug', False)
    departure_index = 0
    print(f'Решаем для {drones_n} дрон{"a" if drones_n == 1 else "ов"}')
    # Определения переменных                                                    https://www.cvxpy.org/tutorial/intro/index.html#vectors-and-matrices
    X = cp.Variable(distance_matrix.shape, boolean=True)                        # Булевая матрица, где 1 означает наличие маршрута между точками i и j
    u = cp.Variable(destinations_n, integer=True)
    # Определение целевой функции
    objective = cp.Minimize(cp.sum(cp.multiply(distance_matrix, X)))
    constraints = get_constraints(X, u, drones_n, destinations_n, len(coords), departure_index)
    # Решение задачи
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)
    if X.value is None:
        raise RuntimeError('Невозможно найти решение')
    # Преобразование решения в маршруты
    X_sol = np.argwhere(X.value==1)
    if debug:
        print('[DEBUG] X:\n', X.value)
        print('[DEBUG] u:\n', u.value)
        print('[DEBUG] X_sol:\n', X_sol)
    print_result(X_sol, coords, departure_index)
    # Вывод длины оптимального маршрута
    optimal_distance = np.sum(np.multiply(distance_matrix, X.value))
    print(f'Длина оптимального маршрута: {np.round(optimal_distance, 2)}км')

    # Calculate per-drone distances
    drone_routes = extract_routes(X_sol, departure_index, drones_n)
    drone_distances = []
    for route in drone_routes:
        dist = sum(distance_matrix[route[i], route[i+1]] for i in range(len(route) - 1))
        drone_distances.append(dist)
    
    if debug:
        print("[DEBUG] Drone distances", drone_distances)
    return optimal_distance, drone_distances

################################################
# Решение задачи целочисленного программирования
################################################
def find_optimal_amount_of_drones(distance_matrix, destinations_n, coords, opt={}):
    lowest_result = float('inf')
    corresponding_i = -1
    departure_index = 0
    arrival_index = 0
    max_cycles = destinations_n if departure_index == arrival_index else destinations_n-1

    for i in range(1, max_cycles):
        result, _ = solve(i, distance_matrix, destinations_n, coords, opt)

        if result < lowest_result:
            lowest_result = result
            corresponding_i = i

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

def run_optimal_solver(data_filepath, opt={}):
    debug = opt.get('debug', False)
    data = get_optimal_solver_input(data_filepath)
    coords = data["coords"]

    n = len(coords)
    C = np.zeros((n,n))

    for i in range(0, n):
        for j in range(0, len(coords)):
            C[i,j] = distance.distance(coords[i], coords[j]).km

    if debug:
        print('[DEBUG] Distance matrix:\n')
        print(np.round(C,1))

    lowest_result, corresponding_i = find_optimal_amount_of_drones(C, n, coords, {'debug': debug})
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
    data = get_single_solver_input(data_filepath)
    coords = data["coords"]
    n = len(coords)
    C = np.zeros((n,n))
    for i in range(0, n):
        for j in range(0, len(coords)):
            C[i,j] = distance.distance(coords[i], coords[j]).km
    result, _ = solve(data["drones_n"], C, n, coords, opt)
    print(f'\nResult: ({np.round(result, 2)})km.')
