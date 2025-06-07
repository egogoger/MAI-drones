import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_surface(X, Y, Z, zlabel):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='plasma')
    ax.set_xlabel('Кол-во дронов')
    ax.set_ylabel('Кол-во точек')
    ax.set_zlabel(zlabel)
    # plt.title(title)
    plt.tight_layout()
    plt.show()

def main():
    # Загрузка и фильтрация данных
    df = pd.read_csv('stats.csv')
    df = df[(df['mode'] == 'solve') & 
            (df['platform'] == 'Windows') & 
            (df['visits_n'] <= 40)].copy()
    df['visits_n'] += 1

    # Группировка
    grouped = df.groupby(['drones_n', 'visits_n'])['seconds']
    stats_df = grouped.agg(std=np.std).reset_index()

    # Поворот в таблицу
    df_pivot = stats_df.pivot(index='visits_n', columns='drones_n', values='std')

    # Сетка координат
    xvals = df_pivot.columns.values
    yvals = df_pivot.index.values
    X, Y = np.meshgrid(xvals, yvals)
    Z = df_pivot.values

    # Построение графика
    # title = f"3D Surface: STD времени (solve / Windows / visits_n ≤ 40)"
    zlabel = "Ст. отклонение (сек)"
    plot_surface(X, Y, Z, zlabel)

if __name__ == '__main__':
    main()
