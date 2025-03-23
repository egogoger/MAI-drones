import json
import os
import numpy as np
from geopy import distance

from solver import find_optimal_amount_of_drones

"""
Reads, validates, and returns structured input data from a JSON file.

Expected JSON structure:
{
    "coords": [[latitude, longitude], ...],
    "debug": bool,
    "departure_index": number
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

    required_keys = ["coords", "debug", "departure_index"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key: {key}")

    if not is_valid_list_of_coords(data["coords"]):
        raise ValueError("Invalid format for 'visits'. Expected list of [latitude, longitude].")

    if not isinstance(data["debug"], bool):
        raise ValueError("Invalid type for 'debug'. Expected boolean.")

    if not isinstance(data["departure_index"], int):
        raise ValueError("Invalid type for 'can_skip'. Expected integer.")

    return data


def main():
    data = read_and_validate_input_from_file("data/optimal.json")
    coords = data["coords"]

    n = len(coords)
    C = np.zeros((n,n))

    for i in range(0, n):
        for j in range(0, len(coords)):
            C[i,j] = distance.distance(coords[i], coords[j]).km

    if data["debug"]:
        print('Distance matrix:\n')
        print(np.round(C,1))

    lowest_result, corresponding_i = find_optimal_amount_of_drones(C, n, coords,
        {'debug': data["debug"], 'departure_index': data["departure_index"]})
    print(f'\nOptimal drones amount: {corresponding_i} ({np.round(lowest_result, 2)})km.')

if __name__ == "__main__":
    main()
