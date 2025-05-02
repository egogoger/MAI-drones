import json
import os
import numpy as np
import cvxpy as cp

from solver_utils import create_index_matrix, generate_random_points, get_distance_matrix, get_vars_and_obj, solve_problem
from utils import evaluate_paths, get_iso_timestamp_for_filename, print_result, save_to_json, write_stats
from constants import MAX_DRONES_N

def get_constraints(X, u, drones_n, coords_n, opt={}):
    debug = opt.get('debug', False)
    forced_paths = opt.get('forced_paths', None)

    ones = np.ones((coords_n, 1))
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
    for i in range(1, coords_n):
        for j in range(1, coords_n):
            if i != j:
                c += [ u[i] - u[j] + 1  <= (coords_n - 1) * (1 - X[i, j]) ]

    # (10) Force required paths
    if forced_paths is not None:
        for from_node, to_node in forced_paths:
            c += [X[from_node, to_node] + X[to_node, from_node] >= 0.999]
            if debug:
                print(f'(10) Forced bidirectional path: {from_node} <-> {to_node}')

    return c

################################################
# Решение задачи целочисленного программирования
################################################
def solve(drones_n, distance_matrix, coords, opt={}):
    debug = opt.get('debug', False)
    forced_paths = opt.get('forced_paths', None)
    departure_i = 0
    print(f'Решаем для {drones_n} дрон{"a" if drones_n == 1 else "ов"}{f'(forced_paths={len(forced_paths)})' if forced_paths else ""}')
    X, u, objective = get_vars_and_obj(distance_matrix)
    constraints = get_constraints(X, u, drones_n, len(coords), opt)
    X_sol, elapsed = solve_problem(objective, constraints, X)

    if debug:
        print('[DEBUG] X:\n', X.value)
        print('[DEBUG] u:\n', u.value)
        print('[DEBUG] X_sol:\n', X_sol)

    write_stats('solve', drones_n, len(distance_matrix)-1, elapsed, forced_paths)

    # Вывод длины оптимального маршрута
    ruta = print_result(X_sol, coords, departure_i, distance_matrix)
    return ruta

def find_optimal_amount_of_drones(coords, max_drones_n, opt={}):
    distance_matrix = get_distance_matrix(coords)
    lowest_result = float('inf')
    corresponding_i = -1
    rutas = []

    for i in range(1, max_drones_n+1):
        ruta = solve(i, distance_matrix, coords, opt)
        rutas.append(ruta)

        if ruta['operation_time'] < lowest_result:
            lowest_result = ruta['operation_time']
            corresponding_i = i
    
    evaluate_paths(rutas)
    save_to_json(rutas, f'results/optimal_{max_drones_n}_{get_iso_timestamp_for_filename()}.json')
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

    if "max_drones_n" in data and not isinstance(data["max_drones_n"], int):
        raise ValueError("Invalid type for 'max_drones_n'. Expected integer.")

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
    max_drones_n = 0
    if random:
        input = get_optimal_random_solver_input(data_filepath)
        coords = generate_random_points(input["coords_n"])
        max_drones_n = input.get('max_drones_n', max(input["coords_n"], MAX_DRONES_N))
    else:
        input = get_optimal_solver_input(data_filepath)
        coords = input["coords"]
        max_drones_n = input.get('max_drones_n', max(len(coords), MAX_DRONES_N))
    if debug:
        print("[DEBUG] Coords", coords)
    lowest_result, corresponding_i = find_optimal_amount_of_drones(coords, max_drones_n, {'debug': debug})
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

"""
Reads, validates, and returns structured input data from a JSON file.

Expected JSON structure:
{
    "coords_n": number,
    "drones_n": number
}
"""
def get_single_random_solver_input(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")

    required_keys = ["coords_n", "drones_n"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key: {key}")

    if not isinstance(data["coords_n"], int):
        raise ValueError("Invalid type for 'coords_n'. Expected integer.")

    if not isinstance(data["drones_n"], int):
        raise ValueError("Invalid type for 'drones_n'. Expected integer.")

    return data

def run_single_solver(data_filepath, opt={}):
    debug = opt.get('debug', False)
    random = opt.get('random', False)
    coords = []
    drones_n = 0
    if random:
        input = get_single_random_solver_input(data_filepath)
        coords = generate_random_points(input["coords_n"])
        drones_n = input["drones_n"]
    else:
        input = get_single_solver_input(data_filepath)
        coords = input["coords"]
        drones_n = input["drones_n"]
        if "forced_paths" in input:
            forced_paths = input["forced_paths"]
            if forced_paths:
                opt["forced_paths"] = forced_paths
    distance_matrix = get_distance_matrix(coords)
    if debug:
        print("[DEBUG] Coords", coords)
        print("[DEBUG] Distance matrix", distance_matrix)
    ruta = solve(drones_n, distance_matrix, coords, opt)
    print(ruta)
