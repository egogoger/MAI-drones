import json
import os
import numpy as np
import cvxpy as cp

from solver_utils import create_index_matrix, generate_random_points, get_distance_matrix, get_vars_and_obj, solve_problem
from utils import print_result, write_stats

"""
Reads, validates, and returns structured input data from a JSON file.

Expected JSON structure:
{
    "arrival": [latitude, longitude],
    "visits": [[latitude, longitude], ...],
    "departures": [[latitude, longitude], ...],
    "can_skip": bool
}
"""
def read_and_validate_input_from_file(filepath):
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

    required_keys = ["arrival", "visits", "departures", "can_skip"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key: {key}")

    if not is_valid_coord(data["arrival"]):
        raise ValueError("Invalid format for 'arrival'. Expected [latitude, longitude].")

    if not is_valid_list_of_coords(data["visits"]):
        raise ValueError("Invalid format for 'visits'. Expected list of [latitude, longitude].")

    if not is_valid_list_of_coords(data["departures"]):
        raise ValueError("Invalid format for 'departures'. Expected list of [latitude, longitude].")

    if not isinstance(data["can_skip"], bool):
        raise ValueError("Invalid type for 'can_skip'. Expected boolean.")

    return data

def get_constraints2(X, u, visits_n, departures_n, coords_n, opt={}):
    can_skip = opt.get('can_skip', False)
    debug = opt.get('debug', False)

    c = []
    ones = np.ones((coords_n, 1))
    arrival_i = 0
    departures_1_i = coords_n-departures_n   # Индекс первой точки отправления
    index_matrix = create_index_matrix(coords_n)

    if debug:
        print('Дебажим constraints на матрице индексов:\n', index_matrix)

    if not can_skip and departures_n > visits_n:
        print("can_skip включён, потому что дронов больше, чем точек посещения")
        can_skip = True

    # (1) Из каждой точки отправления должен вылететь один дрон
    # кроме как в arrival_i (not can_skip) и самих себя
    start = 0 if can_skip else arrival_i+1
    c += [X[departures_1_i:,:] @ ones == 1]
    if debug:
        print('(1) В каждой строке должна быть одна 1:\n',
              index_matrix[departures_1_i:,:])

    # (2) В точку возвращения должно вернуться departures_n дронов
    # При can_skip >= 1 дронов
    tmp_sum = cp.sum(X[arrival_i+1:visits_n+1, arrival_i])
    if can_skip:
        c += [tmp_sum >= 1]
        if debug:
            print('(2) Сумма должна быть >= 1:\n',
                  index_matrix[arrival_i+1:visits_n+1, arrival_i])
    else:
        c += [tmp_sum == departures_n]
        if debug:
            print(f'(2) Сумма должна быть {departures_n}:\n',
                  index_matrix[arrival_i+1:visits_n+1, arrival_i])

    # (3) В каждую точку visits только один вход и только один выход
    c += [X[arrival_i+1:visits_n+1, :] @ ones == 1]                             # В каждой строке матрицы только одна 1
    c += [X[:, arrival_i+1:visits_n+1].T @ ones == 1]                           # В каждой строке матрицы только одна 1
    if debug:
        print(f'(3) В каждой строке должна быть одна 1:\n',
              index_matrix[arrival_i+1:visits_n+1, :])
        print(f'(3) В каждой строке должна быть одна 1:\n',
              index_matrix[:, arrival_i+1:visits_n+1].T)

    # (4) Нет пути из точки в саму себя
    c += [cp.diag(X) == 0]
    if debug:
        print('(4) В каждой ячейке должно быть 0:\n', np.diag(index_matrix))

    # (5) Нельзя из начала сразу в конец
    if not can_skip:
        c += [cp.sum(X[departures_1_i:, 0]) == 0]
        if debug:
            print('(5) В каждой ячейке должно быть 0:\n', index_matrix[departures_1_i:, 0])

    # (6) Нельзя лететь в точки отправления
    c += [cp.sum(X[:, departures_1_i:]) == 0]
    if debug:
        print('(6) В каждой ячейке должно быть 0:\n', index_matrix[:, departures_1_i:])

    # (7) Точки отправления посещаются первыми
    c += [u[departures_1_i:] == 1]

    # (8) Каждая другая точка посещается один раз
    for i in range(0, visits_n+1):
        c += [u[i] >= 2]
        c += [u[i] <= visits_n+2]

    # (9) Порядок посещения точек
    for i in range(arrival_i, departures_1_i):
        for j in range(arrival_i, departures_1_i):
            if i != j:
                c += [ u[i] - u[j] + 1  <= (visits_n + 1) * (1 - X[i, j]) ]

    return c

# This function only solves the task when drones have already departed
def solve_recalc(arrival, visits, departures, opt={}):
    debug = opt.get('debug', False)
    all_coords = np.concatenate(([arrival], visits, departures), axis=0)
    distance_matrix = get_distance_matrix(all_coords)
    if debug:
        print('Distance matrix:\n', distance_matrix)

    X, u, objective = get_vars_and_obj(distance_matrix)
    constraints = get_constraints2(X, u, len(visits), len(departures), len(all_coords), opt)
    X_sol, elapsed = solve_problem(objective, constraints, X)

    if debug:
        print('X:\n', X.value)
        print('u:\n', u.value)
        print('X_sol:\n', X_sol)

    write_stats('recalc', len(departures), len(visits)+1, elapsed)

    departures_indexes = np.arange(1+len(visits), 1+len(visits)+len(departures))
    print_result(X_sol, all_coords, departures_indexes, distance_matrix)

    optimal_distance = np.sum(np.multiply(distance_matrix, X.value))
    print(f'Длина оптимального маршрута: {np.round(optimal_distance, 2)}км')
    return optimal_distance

def main_random(visits_n, departures_n):
    def run_solve_recalc(coords, visits_n, departures_n):
        arrival = coords[0:1][0]
        visits = coords[1:1+visits_n]
        departures = coords[1+visits_n:1+visits_n+departures_n]
        print(visits)
        print(departures)
        solve_recalc(arrival, visits, departures, {'debug': False, 'can_skip': True})
    run_solve_recalc(generate_random_points(1+visits_n+departures_n), visits_n, departures_n)
    del run_solve_recalc

def run_recalc_solver(filepath, opt={}):
    data = read_and_validate_input_from_file(filepath)
    solve_recalc(data["arrival"], data["visits"], data["departures"], {'debug': opt.get("debug", False), 'can_skip': data["can_skip"]})

if __name__ == "__main__":
    # main_random(50, 10)
    run_recalc_solver("data/lost.json")
