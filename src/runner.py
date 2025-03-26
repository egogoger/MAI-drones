from constants import MAX_DRONES_N
from solver import solve
from solver_utils import get_distance_matrix, generate_random_points

def run_solve_tests(start_coords_n=2, max_coords_n=31):
    for coords_n in range(start_coords_n, max_coords_n+1):
        max_drones_n = min(coords_n, MAX_DRONES_N)
        for drones_n in range(1, max_drones_n):
            coords = generate_random_points(coords_n)
            distance_matrix = get_distance_matrix(coords)
            solve(drones_n, distance_matrix, coords)
            print(f"Ran coords_n={coords_n}/{max_coords_n}, drones_n={drones_n}/{max_drones_n-1}")
