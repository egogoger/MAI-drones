import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from geopy import distance

################################################
# Написать оптимальный маршрут
################################################
def show_path(ruta):
    for i in ruta.keys():
        print(i,' => '.join(map(str, ruta[i])))

################################################
# Нарисовать оптимальный маршрут
################################################
def plot_path(ruta, destinations_count, coords):
    # Transforming the coords to the xy plane approximately
    xy_cords = np.zeros((destinations_count, 2))

    for i in range(0, destinations_count):
        xy_cords[i,0] = distance.distance((coords[0][1],0), (coords[i][1],0)).km
        xy_cords[i,1] = distance.distance((0,coords[0][0]), (0,coords[i][0])).km

    # Plotting the coords
    fig, ax = plt.subplots(figsize=(7,3))

    for i in range(destinations_count):
        ax.annotate(str(i), xy=(xy_cords[i,0], xy_cords[i,1]+0.1))

    ax.scatter(xy_cords[:,0],xy_cords[:,1])
    for i in ruta.keys():
        ax.plot(xy_cords[ruta[i],0], xy_cords[ruta[i],1], label = i)
        ax.legend(loc='best')

    plt.show()


################################################
# Создать объект для вывода опт маршрута
################################################
def make_ruta(X_sol, departure_index):
    arrival_index = 0
    ruta = {}
    first_routes_indexes = np.where(X_sol[:,0] == departure_index)[0]
    for i in range(0, len(first_routes_indexes)):
        tmp = X_sol[first_routes_indexes[i], 1]
        ruta['Salesman_' + str(i+1)] = [departure_index, tmp]
        while tmp!=arrival_index:
            tmp = X_sol[np.where(X_sol[:,0] == tmp)[0][0],1]
            ruta['Salesman_' + str(i+1)].append(tmp)
    return ruta


################################################
# Вывести опт маршрут
################################################
def print_result(drones_amount, X_sol, destinations_count, coords, departure_index):
    ruta = make_ruta(X_sol, departure_index)
    plot_path(ruta, destinations_count, coords)
    show_path(ruta)

################################################
# Задание ограничений
################################################
def get_constraints(X, u, drones_amount, destinations_count, coords_count, departure_index):
    ones = np.ones((destinations_count, 1))
    constraints = []
    arrival_index = 0
    # Сумма по строке отправления
    constraints += [X[departure_index,:] @ ones == drones_amount]   # Из точки отправления должно выйти drones_amount дронов
    # Сумма по столбцу прибытия
    constraints += [X[:,arrival_index] @ ones == drones_amount]   # В точку возвращения должно вернуться drones_amount дронов
    # В каждую точку (кроме 0) будет только один вход и только один выход
    if departure_index == 0:
        constraints += [X[1:,:] @ ones == 1]
        constraints += [X[:,1:].T @ ones == 1]

    if departure_index == destinations_count-1:
        constraints += [X[1:departure_index,:] @ ones == 1]
        constraints += [X[:,1:departure_index].T @ ones == 1]

    if departure_index != arrival_index:
        constraints += [X[departure_index, arrival_index] == 0]

    constraints += [cp.diag(X) == 0]                  # Точка не связана сама с собой
    constraints += [u[departure_index] == 1]          # Точка отправления посещается первой

    for point_index in range(coords_count):
        if point_index != departure_index:
            constraints += [u[point_index] >= 2]      # Каждая другая точка посещается один раз
            constraints += [u[point_index] <= coords_count]

    for i in range(1, destinations_count):
        for j in range(1, destinations_count):
            if i != j:
                constraints += [ u[i] - u[j] + 1  <= (destinations_count - 1) * (1 - X[i, j]) ]
    return constraints

################################################
# Решение задачи целочисленного программирования
################################################
def solve(drones_amount, distance_matrix, destinations_count, coords, opt={}):
    debug = opt.get('debug', False)
    departure_index = opt.get('departure_index', 0)
    print(f'Решаем для {drones_amount} дрон{"a" if drones_amount == 1 else "ов"}')
    # Определения переменных                                                    https://www.cvxpy.org/tutorial/intro/index.html#vectors-and-matrices
    X = cp.Variable(distance_matrix.shape, boolean=True)                        # Булевая матрица, где 1 означает наличие маршрута между точками i и j
    u = cp.Variable(destinations_count, integer=True)
    # Определение целевой функции
    objective = cp.Minimize(cp.sum(cp.multiply(distance_matrix, X)))
    constraints = get_constraints(X, u, drones_amount, destinations_count, len(coords), departure_index)
    # Решение задачи
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)
    if X.value is None:
        raise RuntimeError('Невозможно найти решение')
    # Преобразование решения в маршруты
    X_sol = np.argwhere(X.value==1)
    if debug:
        print('X:\n', X.value)
        print('u:\n', u.value)
        print('X_sol:\n', X_sol)
    print_result(drones_amount, X_sol, destinations_count, coords, departure_index)
    # Вывод длины оптимального маршрута
    optimal_distance = np.sum(np.multiply(distance_matrix, X.value))
    print(f'Длина оптимального маршрута: {np.round(optimal_distance, 2)}км')
    return optimal_distance

################################################
# Решение задачи целочисленного программирования
################################################
def find_optimal_amount_of_drones(distance_matrix, destinations_count, coords, opt={}):
    lowest_result = float('inf')
    corresponding_i = -1
    departure_index = opt.get('departure_index', 0)
    arrival_index = 0
    max_cycles = destinations_count if departure_index == arrival_index else destinations_count-1

    for i in range(1, max_cycles):
        result = solve(i, distance_matrix, destinations_count, coords, opt)

        if result < lowest_result:
            lowest_result = result
            corresponding_i = i

    return lowest_result, corresponding_i
