import os
import platform
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import folium

################################################
# Написать оптимальный маршрут
################################################
def print_path(ruta):
    for i in range(0, len(ruta['paths'])):
        print(f'drone_{i+1}',' => '.join(map(str, ruta['paths'][i]['path'])))


def plot_drone_routes(ruta, coords, output_file='drone_routes.html'):
    # Initialize map centered at the first coordinate
    m = folium.Map(location=coords[0], zoom_start=11)

    colors = ['blue', 'green', 'red', 'purple', 'orange', 'darkred', 'cadetblue']
    color_cycle = iter(colors)

    for i in range(0, len(ruta['paths'])):
        drone_id = f'drone_{i+1}'
        info = ruta['paths'][i]
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
    drone_speed = 30 # kph
    arrival_index = 0
    ruta = {'paths': []}
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
        ruta['paths'].append({'path': path, 'distance': dist})
    distances = [float(drone['distance']) for drone in ruta['paths']]
    ruta['total_time'] = sum(distances)/drone_speed*60 # min
    ruta['operation_time'] = max(distances)/drone_speed*60 # min
    return ruta

################################################
# Вывести опт маршрут
################################################
def print_result(X_sol, coords, departures_indexes, distance_matrix):
    ruta = make_ruta(X_sol, departures_indexes, distance_matrix)
    if not os.environ.get('NO_PLOTS'):
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
    total_operational_times = []
    max_operational_times = []
    drone_counts = []

    for ruta in rutas:
        total_operational_times.append(ruta['total_time'])
        max_operational_times.append(ruta['operation_time'])
        drone_counts.append(len(ruta))

    # Calculate ranges and padding
    x_min, x_max = min(total_operational_times), max(total_operational_times)
    y_min, y_max = min(max_operational_times), max(max_operational_times)
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_pad = x_range * 0.05 if x_range > 0 else 5
    y_pad = y_range * 0.05 if y_range > 0 else 5

    # Set limits with padding
    xlim = (max(0, x_min - x_pad), x_max + x_pad)
    ylim = (max(0, y_min - y_pad), y_max + y_pad)

    # Match figure size to aspect
    scale = 0.05
    fig_width = (xlim[1] - xlim[0]) * scale
    fig_height = (ylim[1] - ylim[0]) * scale

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    scatter = ax.scatter(total_operational_times, max_operational_times, c=drone_counts, cmap='viridis', s=100)

    for i, count in enumerate(drone_counts):
        ax.annotate(f'{count} D', (total_operational_times[i], max_operational_times[i]),
                    textcoords="offset points", xytext=(0, 5), ha='center')

    ax.set_xlabel('Σ Время полёта (мин)')
    ax.set_ylabel('Время операции (мин)')
    ax.set_title('Оценка временной эффективности флота дронов')
    ax.grid(True)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()

def write_stats(mode, drones_n, visits_n, seconds):
    with open('stats.csv', 'a') as file:
        file.write(f'{mode},{drones_n},{visits_n},{seconds},{os.cpu_count()},{platform.system()}\n')

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)

def save_to_json(data, filename: str) -> None:
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
        print(f"Data saved to {filename}")
    except (TypeError, IOError) as e:
        print(f"Failed to save data: {e}")

"""
Get the current timestamp in ISO 8601 format, safe for filenames.

Returns:
    str: A timestamp string like '2025-03-26T14-22-05'
"""
def get_iso_timestamp_for_filename() -> str:
    now = datetime.now()
    return now.isoformat(timespec='seconds').replace(":", "-")

