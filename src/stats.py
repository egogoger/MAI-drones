import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Read the CSV file (replace with your actual file path)
    df = pd.read_csv('stats_clean.csv')
    
    # =========================================================
    # 1) 3D Scatter Plot (NO grouping, raw data)
    # =========================================================
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(df['drones_n'], df['visits_n'], df['seconds'])
    ax.set_xlabel('Кол-во дронов')
    ax.set_ylabel('Кол-во точек')
    ax.set_zlabel('Время (сек)')
    # plt.title('3D Scatter (Raw Data)')
    plt.show()

    # =========================================================
    # 2) 3D Surface Plot (ONLY grouping by median here)
    # =========================================================
    # Group by (drones_n, visits_n) and compute the median seconds
    df_agg = df.groupby(['drones_n', 'visits_n'], as_index=False)['seconds'].median()

    # Pivot to form a grid
    df_pivot = df_agg.pivot(index='visits_n', columns='drones_n', values='seconds')

    # Create coordinate mesh
    xvals = df_pivot.columns.values  # Unique drones_n
    yvals = df_pivot.index.values    # Unique visits_n
    X, Y = np.meshgrid(xvals, yvals)
    Z = df_pivot.values             # 2D array of median seconds

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)
    
    ax.set_xlabel('Кол-во дронов')
    ax.set_ylabel('Кол-во точек')
    ax.set_zlabel('Время (сек)')
    # plt.title('3D Surface (Median Aggregated)')
    plt.show()

    # =========================================================
    # 3) 2D "Heatmap" (NO grouping, raw data, color = seconds)
    # =========================================================
    # We'll do a 2D scatter with color scale to represent seconds
    plt.figure()
    sc = plt.scatter(df['drones_n'], df['visits_n'], c=df['seconds'])
    plt.colorbar(sc, label='seconds')
    plt.xlabel('Кол-во дронов')
    plt.ylabel('Кол-во точек')
    # plt.title('2D Scatter with Color = seconds (Raw Data)')
    plt.show()

if __name__ == '__main__':
    main()
