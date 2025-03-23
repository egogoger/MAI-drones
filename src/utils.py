import numpy as np
import matplotlib.pyplot as plt
from geopy import distance

################################################
# Написать оптимальный маршрут
################################################
def print_path(ruta):
    for i in ruta.keys():
        print(i,' => '.join(map(str, ruta[i]['path'])))

################################################
# Нарисовать оптимальный маршрут
################################################
def plot_path(ruta, coords):
    n = len(coords)

    # Transforming the coords to the xy plane approximately
    xy_cords = np.zeros((n, 2))

    for i in range(0, n):
        xy_cords[i,0] = distance.distance((coords[0][1],0), (coords[i][1],0)).km
        xy_cords[i,1] = distance.distance((0,coords[0][0]), (0,coords[i][0])).km

    # Plotting the coords
    fig, ax = plt.subplots(figsize=(17,13))

    for i in range(n):
        ax.annotate(str(i), xy=(xy_cords[i,0], xy_cords[i,1]+0.1))

    ax.scatter(xy_cords[:,0],xy_cords[:,1])
    for i in ruta.keys():
        ax.plot(xy_cords[ruta[i]['path'],0], xy_cords[ruta[i]['path'],1], label = i)
        ax.legend(loc='best')
    
    plt.axis('equal')
    plt.show()

################################################
# Создать объект для вывода опт маршрута
################################################
def make_ruta(X_sol, departures_indexex, distance_matrix):
    arrival_index = 0
    ruta = {}
    first_routes_indexes = np.where(np.isin(X_sol[:, 0], departures_indexex))[0]
    for i in range(0, len(first_routes_indexes)):
        first_route = X_sol[first_routes_indexes[i]]
        tmp = first_route[1]
        path = [first_route[0], tmp]
        while tmp!=arrival_index:
            tmp = X_sol[np.where(X_sol[:,0] == tmp)[0][0],1]
            path.append(tmp)
        dist = 0
        for j in range(len(path) - 1):
            from_node = int(path[j])
            to_node = int(path[j+1])
            dist += distance_matrix[from_node][to_node]
        ruta['drone_' + str(i+1)] = {'path': path, 'distance': dist}
    return ruta

################################################
# Вывести опт маршрут
################################################
def print_result(X_sol, coords, departures_indexes, distance_matrix):
    ruta = make_ruta(X_sol, departures_indexes, distance_matrix)
    plot_path(ruta, coords)
    print_path(ruta)
    return ruta
