import time

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

def mfc(projects, tasks, profits, costs, proy_tasks, actual_tasks, actual_projects, quote, times, profits_records, best_sol, best_profit, best_cost, start_time, output_file):
    while time.time() - start_time <= 45 * 60:  # Limit execution to 30 minutes
        for i in range(projects):
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
                    current_time = time.time() - start_time
                    best_profit = profit
                    best_cost = cost
                    best_sol = latest_actual_projects
                    with open(output_file, 'a') as file:
                        file.write(f"{current_time}, {best_profit}\n")

                # Recursive call
                best_sol, best_profit, best_cost = mfc(projects, tasks, profits, costs, proy_tasks, latest_actual_tasks, latest_actual_projects, new_quote, times, profits_records, best_sol, best_profit, best_cost, start_time, output_file)

        # If no new projects can be added, break the loop
        break

    return best_sol, best_profit, best_cost

def exec(projects, tasks, budget, profits, costs, proy_tasks):
    start_time = time.time()
    best_sol, best_profit, best_cost = set(), 0, 0
    times, profits_records = [], []
    output_file = "1-2024-45-min-res.txt"

    best_sol, best_profit, best_cost = mfc(projects, tasks, profits, costs, proy_tasks, set(), set(), budget, times, profits_records, best_sol, best_profit, best_cost, start_time, output_file)

    return best_sol, best_profit, best_cost, times, profits_records

if __name__ == "__main__":
    projects, tasks, budget, profits, costs, proy_tasks = read_file("1-2024.txt")
    best_sol, best_profit, best_cost, times, profits_records = exec(projects, tasks, budget, profits, costs, proy_tasks)
