import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Define the objective functions
def f1(x):
    return sum(xi**2 - 10 * np.cos(2 * np.pi * xi) + 10 for xi in x)

def f2(x):
    return sum(100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(len(x)-1))

def f3(x):
    return -sum(np.sin(xi) * (np.sin(i * xi**2 / np.pi))**20 for i, xi in enumerate(x, 1))

# Genetic Algorithm (GA) functions
def initialize_population(pop_size, dim, bounds):
    return np.random.rand(pop_size, dim) * (bounds[1] - bounds[0]) + bounds[0]

def evaluate_population(population, func):
    return np.array([func(ind) for ind in population])

def tournament_selection(population, fitness, k=3):
    selected = []
    pop_size = len(population)
    for _ in range(pop_size):
        aspirants = np.random.choice(pop_size, k, replace=False)
        best = aspirants[np.argmin(fitness[aspirants])]
        selected.append(population[best])
    return np.array(selected)

def crossover(parent1, parent2, crossover_rate):
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, len(parent1))
        return np.concatenate((parent1[:point], parent2[point:]))
    return parent1

def mutate(individual, mutation_rate, bounds):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = np.random.uniform(bounds[0], bounds[1])
    return individual

def evolve_population(population, fitness, mutation_rate, crossover_rate, bounds):
    new_population = []
    selected_population = tournament_selection(population, fitness)
    for i in range(0, len(population), 2):
        parent1 = selected_population[i]
        parent2 = selected_population[i + 1]
        child1 = crossover(parent1, parent2, crossover_rate)
        child2 = crossover(parent2, parent1, crossover_rate)
        new_population.append(mutate(child1, mutation_rate, bounds))
        new_population.append(mutate(child2, mutation_rate, bounds))
    return np.array(new_population)

# Local Search function (Scatter Search)
def local_search(individual, func, bounds, max_iter=500, alpha=0.05):
    best_individual = individual.copy()
    best_fitness = func(best_individual)
    for _ in range(max_iter):
        gradient = np.array([best_fitness - func(np.clip(best_individual + alpha * unit_vector, bounds[0], bounds[1])) for unit_vector in np.eye(len(individual))])
        new_individual = best_individual - alpha * gradient
        new_fitness = func(new_individual)
        if new_fitness < best_fitness:
            best_individual = new_individual
            best_fitness = new_fitness
        else:
            break
    return best_individual

def scatter_search(population, func, bounds, max_iter=100, alpha=0.01):
    for i in range(len(population)):
        population[i] = local_search(population[i], func, bounds, max_iter, alpha)
    return population

def run_ga(func, dim, bounds, pop_size, max_generations, mutation_rate=0.05, crossover_rate=0.3):
    population = initialize_population(pop_size, dim, bounds)
    population = scatter_search(population, func, bounds)  # Apply local search after initialization
    fitness = evaluate_population(population, func)
    best_fitness = np.min(fitness)
    best_index = np.argmin(fitness)
    best_solution = population[best_index]
    convergence = [(0, best_fitness, 0)]
    start_time = time.time()
    for generation in range(1, max_generations + 1):
        population = evolve_population(population, fitness, mutation_rate, crossover_rate, bounds)
        fitness = evaluate_population(population, func)
        current_best_fitness = np.min(fitness)
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_index = np.argmin(fitness)
            best_solution = population[best_index]
        elapsed_time = time.time() - start_time
        convergence.append((elapsed_time, best_fitness, generation))
    return best_solution, best_fitness, generation, convergence

# Parameters for GA
pop_size = 100  # Increased population size for better exploration
max_generations = 500  # Increased generations for better optimization
mutation_rate = 0.01 #valor inicial: 0.01. 0.05
crossover_rate = 0.7 #valor inicial: 0.7, 0.5, 1 conviene mas 0.7

def execute_ga_multiple_times(func, dim, bounds, pop_size, max_generations, num_executions=10):
    results = []
    convergence_data = []
    for _ in range(num_executions):
        best_solution, best_fitness, generation, convergence = run_ga(func, dim, bounds, pop_size, max_generations, mutation_rate, crossover_rate)
        cpu_time = convergence[-1][0]  # Last recorded CPU time
        results.append((best_solution, best_fitness, generation, cpu_time, convergence))
        convergence_data.append(convergence)
    return results, convergence_data

def summarize_results(results):
    avg_fitness = np.mean([result[1] for result in results])
    avg_cpu_time = np.mean([result[3] for result in results])
    return avg_fitness, avg_cpu_time

def plot_convergence(convergence_data, func_names):
    fig, axs = plt.subplots(1, len(convergence_data), figsize=(20, 6))
    if len(convergence_data) == 1:
        axs = [axs]  # Convert to list for consistency
    for ax, convergence, func_name in zip(axs, convergence_data, func_names):
        for conv in convergence[:2]:  # Plot only the first two runs
            cpu_times, best_fitness, _ = zip(*conv)
            ax.plot(cpu_times, best_fitness, label='Run')
        ax.set_xlabel('CPU Time (s)')
        ax.set_ylabel('Best Fitness')
        ax.set_title(f'Convergence of GA for {func_name}')
        ax.legend()
        
        # Ajustar el formato del eje y para no utilizar notación científica
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='plain', axis='y')
        ax.yaxis.get_major_formatter().set_scientific(False)  # Desactiva notación científica
        ax.yaxis.get_major_formatter().set_useOffset(False)   # Desactiva el offset

    plt.tight_layout()
    plt.show()

def save_results_to_txt(func_name, results, initial_parameters, best_result, best_iteration, avg_fitness, avg_cpu_time):
    filename = f'results_{func_name}.txt'
    with open(filename, 'a') as file:
        file.write('--------------------------------------------------\n')
        file.write('Parametros iniciales\n')
        file.write(f'{initial_parameters}\n')
        file.write('--------------------------------------------------\n')
        file.write('Resultados de las 10 ejecuciones:\n')
        for result in results:
            best_solution_str = ', '.join(map(str, result[0]))  # Convert array to string
            file.write(f'Best solution: {best_solution_str}\n')
            file.write(f'Best fitness: {result[1]}\n')
            file.write(f'Iteration: {result[2]}\n')
            file.write(f'CPU time: {result[3]:.4f}s\n')
            file.write('--------------------------------------------------\n')
        file.write('Resumen de resultados:\n')
        file.write(f'Average best fitness: {avg_fitness}\n')
        file.write(f'Average CPU time: {avg_cpu_time:.4f} s\n')
        best_solution_str = ', '.join(map(str, best_result[0]))  # Convert array to string
        file.write(f'Best overall solution: {best_solution_str}\n')
        file.write(f'Best overall fitness: {best_result[1]}\n')
        file.write(f'Best overall iteration: {best_iteration}\n')
        file.write(f'CPU time for best overall solution: {best_result[3]:.4f} s\n')

functions = [(f1, 10, (-5.12, 5.12), 'f1'), (f2, 20, (-30, 30), 'f2'), (f3, 5, (0, np.pi), 'f3')]

all_convergence_data = []
func_names = []

for func, dim, bounds, func_name in functions:
    initial_parameters = f'Function: {func_name}, Dimension: {dim}, Bounds: {bounds}, Population Size: {pop_size}, Max Generations: {max_generations}, Mutation Rate: {mutation_rate}, Crossover Rate: {crossover_rate}'
    print(f"Executing GA for {func_name} 10 times...\n")
    results, convergence_data = execute_ga_multiple_times(func, dim, bounds, pop_size, max_generations)
    # Extract fitness values and find the index of the best result
    fitness_values = [result[1] for result in results]
    best_index = np.argmin(fitness_values)
    best_result = results[best_index]
    best_convergence = convergence_data[best_index]
    best_iteration = next(iteration for time, fitness, iteration in best_convergence if fitness == best_result[1])
    
    # Calculate average fitness and CPU time
    avg_fitness, avg_cpu_time = summarize_results(results)
    
    # Store the results in TXT files
    save_results_to_txt(func_name, results, initial_parameters, best_result, best_iteration, avg_fitness, avg_cpu_time)
    
    # Append convergence data for plotting
    all_convergence_data.append(convergence_data)
    func_names.append(func_name)

# Plot all convergence data in a single figure
plot_convergence(all_convergence_data, func_names)
