import time# type: ignore
import matplotlib.pyplot as plt # type: ignore
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

def mfc_recursivo(projects, tasks, profits, costs, proy_tasks, actual_tasks, actual_projects, quote, times, profits_records, best_sol, best_profit, best_cost, start_time, sorted_project_indices, interval=5, output_file=None):
    last_recorded_time = start_time

    while time.time() - start_time <= 45 * 60:  # Limit execution to 60 minutes
        current_time = time.time()
        elapsed_time = current_time - start_time

        if current_time - last_recorded_time >= interval:
            times.append(elapsed_time)
            profits_records.append(best_profit)
            last_recorded_time = current_time

        for i in sorted_project_indices:
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

                best_sol, best_profit, best_cost = mfc_recursivo(projects, tasks, profits, costs, proy_tasks, latest_actual_tasks, latest_actual_projects, new_quote, times, profits_records, best_sol, best_profit, best_cost, start_time, sorted_project_indices, output_file=output_file)

        # If no new projects can be added, break the loop
        break

    return best_sol, best_profit, best_cost


def exec(projects, tasks, budget, profits, costs, proy_tasks, sorted_project_indices):
    output_file = "output.txt"
    # Limpia el archivo de salida si ya existe
    open(output_file, 'w').close()
    start_time = time.time()
    best_sol, best_profit, best_cost = set(), 0, 0
    times, profits_records = [], []

    best_sol, best_profit, best_cost = mfc_recursivo(projects, tasks, profits, costs, proy_tasks, set(), set(), budget, times, profits_records, best_sol, best_profit, best_cost, start_time, sorted_project_indices, output_file=output_file)

    return best_sol, best_profit, best_cost, times, profits_records


def exec_kmeans_task_similarity(proy_tasks):
    # proy_tasks ya es una lista de listas con vectores binarios de tareas requeridas por cada proyecto
    kmeans = KMeans(n_clusters=4, random_state=0).fit(proy_tasks)
    clusters = kmeans.labels_
    return clusters

def sort_projects_by_cluster(profits, clusters):
    cluster_dict = {}
    for i, cluster in enumerate(clusters):
        if cluster not in cluster_dict:
            cluster_dict[cluster] = []
        cluster_dict[cluster].append((profits[i], i))
    
    # Ordenar cada lista de proyectos en cluster_dict por ganancia decreciente
    for cluster in cluster_dict:
        cluster_dict[cluster].sort(reverse=True, key=lambda x: x[0])
    
    # Ordenar los clusters por la ganancia promedio de sus proyectos
    sorted_clusters = sorted(cluster_dict.items(), key=lambda x: sum(proj[0] for proj in x[1])/len(x[1]), reverse=True)
    
    # Aplanar la lista de proyectos ordenados
    sorted_project_indices = [proj[1] for cluster in sorted_clusters for proj in cluster[1]]
    return sorted_project_indices

if __name__ == "__main__":
    projects, tasks, budget, profits, costs, proy_tasks = read_file("2-2024.txt")

    # best_sol, best_profit, best_cost, times, profits_records = exec(projects, tasks, budget, profits, costs, proy_tasks)
    # Ejecutar K-Means basado en la similitud de las tareas de los proyectos
    clusters = exec_kmeans_task_similarity(proy_tasks)
    sorted_project_indices = sort_projects_by_cluster(profits, clusters)
    
    best_sol, best_profit, best_cost, times, profits_records = exec(projects, tasks, budget, profits, costs, proy_tasks, sorted_project_indices)
    



    print(f"La mejor solucion es {best_sol}, con ganancia {best_profit} y costo {best_cost}")
    plt.plot(times, profits_records)
    plt.xlabel('Time (s)')
    plt.ylabel('Profit')
    plt.title('Profit Evolution Over Time')
    plt.show()

##### EXPLICACION:

# Elección del Cluster Inicial
# La elección del cluster con el que comenzar la selección de proyectos se basa en el valor promedio de las ganancias de los proyectos en cada cluster. El criterio es seleccionar primero el cluster cuyos proyectos, en promedio, tienen las mayores ganancias. Esto se hace suponiendo que los proyectos con mayores ganancias contribuirán más al objetivo de maximizar la ganancia total, aunque también podría considerarse el costo asociado si se quiere ajustar la estrategia para tener en cuenta el presupuesto disponible.

# Implementación en sort_projects_by_cluster
# En la función sort_projects_by_cluster, primero agrupamos los proyectos en un diccionario cluster_dict donde cada clave es un índice de cluster y cada valor es una lista de tuplas, con cada tupla representando la ganancia y el índice de un proyecto dentro de ese cluster:

# python
# Copy code
# cluster_dict = {}
# for i, cluster in enumerate(clusters):
#     if cluster not in cluster_dict:
#         cluster_dict[cluster] = []
#     cluster_dict[cluster].append((profits[i], i))
# Luego, ordenamos cada lista de proyectos dentro de cada cluster por su ganancia de manera descendente:

# python
# Copy code
# for cluster in cluster_dict:
#     cluster_dict[cluster].sort(reverse=True, key=lambda x: x[0])
# Finalmente, ordenamos los clusters mismos según la ganancia promedio de los proyectos dentro de cada cluster, de mayor a menor:

# python
# Copy code
# sorted_clusters = sorted(cluster_dict.items(), key=lambda x: sum(proj[0] for proj in x[1])/len(x[1]), reverse=True)
# Esto resulta en que los proyectos de clusters con mayores ganancias promedio se evalúan antes que los de clusters con ganancias promedio menores.

# Orden de los Proyectos Dentro de Cada Cluster
# Una vez definidos los clusters y el orden en que se abordarán, el siguiente paso es determinar el orden de los proyectos dentro de cada cluster. Como mencionado, dentro de cada cluster los proyectos se ordenan de manera descendente según su ganancia individual. Esta decisión se basa en la suposición de que dar prioridad a proyectos con mayores ganancias individuales podría llevar más rápidamente a una solución que maximice las ganancias totales, especialmente bajo la restricción de un presupuesto fijo.
