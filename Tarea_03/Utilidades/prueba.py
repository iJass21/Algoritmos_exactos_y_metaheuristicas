import numpy as np
import time
import matplotlib.pyplot as plt

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

def run_ga(func, dim, bounds, pop_size, max_generations, mutation_rate=0.01, crossover_rate=0.7):
    population = initialize_population(pop_size, dim, bounds)
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
mutation_rate = 0.01
crossover_rate = 0.7

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

def plot_convergence(convergence_data, func_name):
    fig, ax = plt.subplots(figsize=(7, 6))
    for convergence in convergence_data[:2]:  # Plot only the first two runs
        cpu_times, best_fitness, _ = zip(*convergence)
        ax.plot(cpu_times, best_fitness, label='Run')
    ax.set_xlabel('CPU Time (s)')
    ax.set_ylabel('Best Fitness')
    ax.set_title(f'Convergence of GA for {func_name}')
    plt.legend()
    plt.show()

# Execute GA for each function 10 times and show results
functions = [(f1, 10, (-5.12, 5.12), 'f1'), (f2, 20, (-30, 30), 'f2'), (f3, 5, (0, np.pi), 'f3')]

for func, dim, bounds, func_name in functions:
    print(f"Executing GA for {func_name} 10 times...\n")
    results, convergence_data = execute_ga_multiple_times(func, dim, bounds, pop_size, max_generations)
    # Extract fitness values and find the index of the best result
    fitness_values = [result[1] for result in results]
    best_index = np.argmin(fitness_values)
    best_result = results[best_index]
    best_convergence = convergence_data[best_index]
    best_iteration = next(iteration for time, fitness, iteration in best_convergence if fitness == best_result[1])
    print(f"Best result for {func_name}: \nBest solution: {best_result[0]}\n Best fitness: {best_result[1]} \nIteration: {best_iteration} \nCPU time: {best_result[3]:.4f}s \n")
    avg_fitness, avg_cpu_time = summarize_results(results)
    print(f"Average best fitness for {func_name}: {avg_fitness} \n")
    print(f"Average CPU time for {func_name}: {avg_cpu_time:.4f}s \n")
    plot_convergence(convergence_data, func_name)