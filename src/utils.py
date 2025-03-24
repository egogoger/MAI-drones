import numpy as np
import matplotlib.pyplot as plt
from geopy import distance
import folium

################################################
# Написать оптимальный маршрут
################################################
def print_path(ruta):
    for i in ruta.keys():
        print(i,' => '.join(map(str, ruta[i]['path'])))


def plot_drone_routes(ruta, coords, output_file='drone_routes.html'):
    # Initialize map centered at the first coordinate
    m = folium.Map(location=coords[0], zoom_start=11)

    colors = ['blue', 'green', 'red', 'purple', 'orange', 'darkred', 'cadetblue']
    color_cycle = iter(colors)

    for i, (drone_id, info) in enumerate(ruta.items()):
        path_indices = info['path']
        path_coords = [coords[i] for i in path_indices]
        color = next(color_cycle, 'gray')  # fallback to gray if colors run out

        # Add a polyline for this drone
        folium.PolyLine(path_coords, color=color, weight=2.5, opacity=1, popup=f"{drone_id} route").add_to(m)

        # Add markers
        for step, coord in enumerate(path_coords):
            marker_popup = f"{drone_id} - Step {step}<br>Coord Index: {path_indices[step]}"
            if 'distance' in info and step == 0:
                marker_popup += f"<br>Total Distance: {info['distance']:.2f}"

            folium.Marker(
                location=coord,
                popup=marker_popup,
                icon=folium.Icon(color='red' if step == 0 else color)
            ).add_to(m)

    # Save and return the map
    m.save(output_file)
    return m


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
    print_path(ruta)
    plot_drone_routes(ruta, coords)
    return ruta

"""
Plots total distance vs max operational time for different drone data scenarios.

Parameters:
- all_drone_data: List of dicts, where each dict represents a drone_data for a given scenario.
                    Format: {'drone_1': {'distance': float, 'path': [...]}, ...}
"""
def evaluate_paths(rutas):
    drone_speed = 30 # kph
    total_operational_times = []
    max_operational_times = []
    drone_counts = []

    for scenario in rutas:
        distances = [float(drone['distance']) for drone in scenario.values()]
        total_distance = sum(distances)/drone_speed*60 # min
        max_time = max(distances)/drone_speed*60 # min
        num_drones = len(scenario)

        total_operational_times.append(total_distance)
        max_operational_times.append(max_time)
        drone_counts.append(num_drones)

    fig, ax = plt.subplots()
    ax.scatter(total_operational_times, max_operational_times, c=drone_counts, cmap='viridis', s=100)

    for i, count in enumerate(drone_counts):
        ax.annotate(f'{count} D', (total_operational_times[i], max_operational_times[i]), textcoords="offset points", xytext=(0,5), ha='center')

    ax.set_xlabel('Total Fleet Distance [min]')
    ax.set_ylabel('Max Operational Time [min]')
    ax.set_title('Drone Deployment Trade-off: Distance vs Time')
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    ax.set_ylim(bottom=0)
    plt.show()
