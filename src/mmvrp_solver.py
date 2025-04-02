import os
import time
import numpy as np
import cvxpy as cp

from solver import get_single_solver_input
from solver_utils import get_distance_matrix
from utils import plot_drone_routes, print_path, print_result, write_stats

def solve_problem(objective, constraints, X):
    problem = cp.Problem(objective, constraints)
    start = time.time()
    problem.solve(verbose=False)
    end = time.time()
    elapsed = round(end-start, 3)

    if X.value is None:
        raise ValueError("Проблема не решена. Попробуй другой солвер или проверь ограничения.")

    # Преобразуем X в список маршрутов для каждого дрона
    X_sol = []
    for k in range(X.shape[0]):
        X_k = np.round(X.value[k]).astype(int)
        X_sol.append(X_k)

    return X_sol, elapsed

def extract_route(X_k, start_node):
    """ Извлекает маршрут из булевой матрицы X_k, начиная с узла start_node """
    n = X_k.shape[0]
    route = [start_node]
    current = start_node
    visited = set()

    while True:
        next_node = None
        for j in range(n):
            if X_k[current, j] > 0.5 and j not in visited:
                next_node = j
                break
        if next_node is None or next_node == start_node:
            route.append(start_node)
            break
        route.append(next_node)
        visited.add(current)
        current = next_node

    return route

def compute_path_distance(route, distance_matrix):
    dist = 0.0
    for i in range(len(route) - 1):
        dist += distance_matrix[route[i], route[i + 1]]
    return dist

def solve_mmvrp(drones_n, distance_matrix, coords, opt={}):
    debug = opt.get('debug', False)
    coords_n = len(coords)
    departure_i = 0

    print(f'Решаем Min-Max VRP для {drones_n} дрон{"a" if drones_n == 1 else "ов"}')

    # Получение переменных, цели и ограничений
    X, u, T, objective, constraints = get_vars_and_obj_and_constraints(distance_matrix, drones_n, coords_n, opt)

    # Решение задачи
    X_sol, elapsed = solve_problem(objective, constraints, X)

    if debug:
        print('[DEBUG] X (маршруты):\n', X.value)
        print('[DEBUG] u (порядки):\n', u.value)
        print('[DEBUG] X_sol (решение):\n', X_sol)
        print('[DEBUG] T (максимальный путь):\n', T.value)

    write_stats('mmvrp', drones_n, len(distance_matrix)-1, elapsed)

    paths = []
    max_distance = 0.0

    for k, X_k in enumerate(X_sol):
        route = extract_route(X_k, departure_i)
        distance = compute_path_distance(route, distance_matrix)
        paths.append({
            "path": [np.int64(i) for i in route],
            "distance": np.float64(distance)
        })
        max_distance = max(max_distance, distance)

    ruta = {
        "paths": paths,
        "total_time": elapsed,
        "operation_time": np.float64(max_distance)
    }

    if not os.environ.get('NO_PLOTS'):
        print_path(ruta)
        plot_drone_routes(ruta, coords)

    return ruta


def get_vars_and_obj_and_constraints(distance_matrix, drones_n, coords_n, opt={}):
    X = cp.Variable((drones_n, coords_n, coords_n), boolean=True)
    u = cp.Variable((drones_n, coords_n), integer=True)
    T = cp.Variable()

    constraints = []

    departure_i = 0
    arrival_i = 0  # стартовая/конечная точка

    # (1) Каждый дрон стартует и возвращается в 0
    for k in range(drones_n):
        constraints += [
            cp.sum(X[k, departure_i, 1:]) == 1,
            cp.sum(X[k, 1:, arrival_i]) == 1
        ]

    # (2) Из 0 вылетает ровно drones_n дронов, и они возвращаются в 0
    constraints += [
        cp.sum(X[:, departure_i, :]) == drones_n,
        cp.sum(X[:, :, arrival_i]) == drones_n
    ]

    # (3) Каждая точка (кроме 0) посещается ровно 1 раз
    for j in range(1, coords_n):
        constraints += [
            cp.sum(X[:, :, j]) == 1,
            cp.sum(X[:, j, :]) == 1
        ]

    # (4) Дроны не посещают точку более одного раза и не летают в саму себя
    for k in range(drones_n):
        constraints += [cp.sum(cp.diag(X[k])) == 0]

    # (5) Если дрон входит в точку — он из неё и выходит
    for k in range(drones_n):
        for j in range(1, coords_n):
            constraints += [
                cp.sum(X[k, :, j]) == cp.sum(X[k, j, :])
            ]

    # (6) MTZ-подобные ограничения для исключения субтуров
    for k in range(drones_n):
        constraints += [u[k, departure_i] == 1]
        for i in range(1, coords_n):
            constraints += [u[k, i] >= 2, u[k, i] <= coords_n]
        for i in range(1, coords_n):
            for j in range(1, coords_n):
                if i != j:
                    constraints += [
                        u[k, i] - u[k, j] + 1 <= (coords_n - 1) * (1 - X[k, i, j])
                    ]

    # (7) Ограничения на максимальный маршрут
    drone_distances = [
        cp.sum(cp.multiply(distance_matrix, X[k])) for k in range(drones_n)
    ]
    for d in drone_distances:
        constraints.append(d <= T)

    # Цель: минимизировать максимальное расстояние
    objective = cp.Minimize(T)

    return X, u, T, objective, constraints

def run_mmvrp_solver(filepath, opt={}):
    input = get_single_solver_input(filepath)
    distance_matrix = get_distance_matrix(input["coords"])
    ruta = solve_mmvrp(input["drones_n"], distance_matrix, input["coords"], opt)
    print(ruta)
