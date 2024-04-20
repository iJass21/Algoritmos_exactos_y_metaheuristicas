import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

def read_file(filename):
    with open(filename, "r") as file:
        n_projects, n_tasks, budget = map(int, [file.readline() for _ in range(3)])
        profits, costs = [list(map(int, file.readline().split())) for _ in range(2)]
        projects_tasks = [list(map(int, file.readline().split())) for _ in range(n_projects)]
    return n_projects, n_tasks, budget, profits, costs, projects_tasks

def check_constraints(i, proy_tasks, completed_tasks, quote, costs):
    required_tasks = {j for j, assigned in enumerate(proy_tasks[i]) if assigned == 1 and j not in completed_tasks}
    total_project_cost = sum(costs[j] for j in required_tasks)

    if quote >= total_project_cost:
        return True, total_project_cost, required_tasks
    return False, 0, set()

def mfc(projects, tasks, profits, costs, proy_tasks, actual_tasks, actual_projects, quote, times, profits_records, best_sol, best_profit, best_cost, start_time, interval=5, ordered_projects=None, output_file=None):
    last_recorded_time = start_time

    while time.time() - start_time <= 35 * 60:  # tiempo limite de ejecución
        current_time = time.time()
        elapsed_time = current_time - start_time

        if current_time - last_recorded_time >= interval:
            times.append(elapsed_time)
            profits_records.append(best_profit)
            last_recorded_time = current_time

        for i in ordered_projects:
            if i in actual_projects:
                continue
            is_viable, total_project_cost, p_tasks = check_constraints(i, proy_tasks, actual_tasks, quote, costs)
            if is_viable:
                new_quote = quote - total_project_cost
                latest_actual_projects = actual_projects | {i}
                latest_actual_tasks = actual_tasks | p_tasks
                profit = sum(profits[j] for j in latest_actual_projects)
                cost = sum(costs[j] for j in latest_actual_tasks)

                if profit > best_profit or (profit == best_profit and cost < best_cost):
                    best_profit = profit
                    best_cost = cost
                    best_sol = latest_actual_projects
                    # Guardar en el archivo
                    if output_file:
                        with open(output_file, "a") as f:
                            f.write(f"{elapsed_time}, {best_profit}\n")

                # Llamada recursiva
                best_sol, best_profit, best_cost = mfc(projects, tasks, profits, costs, proy_tasks, latest_actual_tasks, latest_actual_projects, new_quote, times, profits_records, best_sol, best_profit, best_cost, start_time, interval, ordered_projects, output_file=output_file)

        break  # Para el while loop una vez que todos los proyectos fueron procesados.

    return best_sol, best_profit, best_cost

def preprocess_and_order_projects(projects_tasks, profits, costs, n_projects):
    # Calcular tareas por proyecto
    tasks_per_project = [sum(tasks) for tasks in projects_tasks]

    # K-Means Clustering
    kmeans = KMeans(n_clusters=4, random_state=0).fit(np.array(tasks_per_project).reshape(-1, 1))
    clusters = {i: [] for i in range(4)}
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(i)

    # Ordenar proyectos para cada cluster por la relacion costo/ganancia
    for cluster in clusters:
        clusters[cluster].sort(key=lambda x: (profits[x] / sum(costs[j] for j in range(len(projects_tasks[x])) if projects_tasks[x][j] == 1), -tasks_per_project[x]), reverse=True)

    # Ordenar los clusters por el numero promedio de tareas (descendiente)
    cluster_order = sorted(clusters.keys(), key=lambda x: -np.mean([tasks_per_project[i] for i in clusters[x]]))
    ordered_projects = [project for cluster in cluster_order for project in clusters[cluster]]
    return ordered_projects

def exec(projects, tasks, budget, profits, costs, proy_tasks):
    output_file = "output.txt"  # Nombre del archivo de salida
    # Limpia el archivo de salida si ya existe
    open(output_file, 'w').close()
    
    ordered_projects = preprocess_and_order_projects(proy_tasks, profits, costs, projects)
    start_time = time.time()
    best_sol, best_profit, best_cost = set(), 0, 0
    times, profits_records = [], []
    best_sol, best_profit, best_cost = mfc(projects, tasks, profits, costs, proy_tasks, set(), set(), budget, times, profits_records, best_sol, best_profit, best_cost, start_time, ordered_projects=ordered_projects, output_file=output_file)
    return best_sol, best_profit, best_cost, times, profits_records

if __name__ == "__main__":
    projects, tasks, budget, profits, costs, proy_tasks = read_file("2-2024.txt")
    best_sol, best_profit, best_cost, times, profits_records = exec(projects, tasks, budget, profits, costs, proy_tasks)
