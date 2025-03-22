import numpy as np
from geopy import distance

from solver import find_optimal_amount_of_drones

def main():
    ################################################
    # Дано
    ################################################
    cities = [(-12.059296, -76.975893),
            (-12.079575, -77.009686),
            (-12.087303, -76.996620),
            (-12.084391, -76.975651),
            (-12.063603, -76.960483),
            (-12.056762, -77.014452),
            (-12.011531, -77.002383)]

    ################################################
    # Строим матрицу расстояний
    ################################################
    n = len(cities)
    C = np.zeros((n,n))

    for i in range(0, n):
        for j in range(0, len(cities)):
            C[i,j] = distance.distance(cities[i], cities[j]).km

    # Showing distance matrix
    print('Матрица расстояний:\n')
    print(np.round(C,1))

    lowest_result, corresponding_i = find_optimal_amount_of_drones(C, n, cities,
        {'debug': False, 'departure_index': 6})
    print(f'\nОптимальное число дронов: {corresponding_i} ({np.round(lowest_result, 2)})км.')

if __name__ == "__main__":
    main()
