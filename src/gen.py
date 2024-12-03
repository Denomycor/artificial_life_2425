from functools import reduce
from typing import OrderedDict
from organism import organism
from simulation import simulation
from utils import vec2
import torch


# Get the fitness value of a population
def fitness(sim) -> float:
    return reduce(lambda acc, e: acc + e.sickness, sim.organism_list, 0)


# TODO: Cross two different genes to create a new one
def cross_genes(st: OrderedDict, nd: OrderedDict) -> OrderedDict:
    child = OrderedDict()
    for key in st.keys():
        if torch.rand(1) < 0.5: # Randomly select gene from either parent
            child[key] = st[key]
        else:
            child[key] = nd[key]
    return child


# TODO: Create a new gene which is a mutation of the parameter gene
def mutate_genes(gene: OrderedDict, mutation_rate: float = 0.1) -> OrderedDict:
    mutated_gene = OrderedDict()
    for key, value in gene.items():
        if "weight" in key or "bias" in key:
            if torch.rand(1).item() < mutation_rate:  # Apply mutation
                mutation = torch.randn_like(value) * 0.1  # Small random perturbation
                mutated_gene[key] = value + mutation
            else:
                mutated_gene[key] = value
        else:
            mutated_gene[key] = value
    return mutated_gene

# TODO: From the population and it's fitness select pairs of organisms to generate offspring
def selection_tournament(population: list[simulation], fit: list[float], tournament_size: int = 3) -> list[tuple[organism, organism]]:
    selected_pairs = []
    for _ in range(len(population) // 2):  # Create pairs
        # Select first parent
        candidates = torch.randperm(len(population))[:tournament_size]
        best_candidate = max(candidates, key=lambda idx: fit[idx])
        parent1 = population[best_candidate].organism_list[0]  # Assume 1 organism per simulation

        # Select second parent
        candidates = torch.randperm(len(population))[:tournament_size]
        best_candidate = max(candidates, key=lambda idx: fit[idx])
        parent2 = population[best_candidate].organism_list[0]

        selected_pairs.append((parent1, parent2))
    return selected_pairs


# Generate a random set of weights and bias for a given neural network
def generate_random_gene(nn_arch: list[int]) -> OrderedDict:
    out = OrderedDict()
    for i in range(len(nn_arch)-1):
        st = nn_arch[i]
        nd = nn_arch[i+1]
        out["l"+ str(i+1)+".weight"] = torch.rand(nd, st)
        out["l"+ str(i+1)+".bias"] = torch.rand(nd)
    return out


# Create the initial state of the first generation
def start_genetic_alg(population_size: int) -> list[simulation]:
    grid_size = vec2(20,20)
    organism_population = 50
    max_steps = 20

    population = []
    for _ in range(population_size):
        sim = simulation(grid_size, organism_population, max_steps)
        sim.random_initial_state()
        population.append(sim)
    return population


# TODO: Work in progress. Main loop of the genetic algorithm
def run_genetic_alg(generations: int, population_size: int):
    # Initialize the population
    population = start_genetic_alg(population_size)
    
    
    for gen in range(generations):
        print(f"Generation {gen + 1}/{generations}")
        
        # Evaluate the fitness of the population
        fit = []
        for individual in population:
            individual.run()
            fit.append(fitness(individual))
            
        # Check that organisms exist in the population
        if not population[0].organism_list:
            raise ValueError("Population has no organisms! Check simulation initialization.")
            
        # Select parents for the next generation
        selected_pairs = selection_tournament(population, fit)
        
        # Crossover and mutation
        new_population = []
        for parent1, parent2 in selected_pairs:
            child_gene = cross_genes(parent1.get_neural_genes(), parent2.get_neural_genes())
            child_gene = mutate_genes(child_gene)
            
            # Create new simulation with the child gene
            sim = simulation(vec2(20,20), 50, 20)
            sim.organism_list[0].set_neural_genes(child_gene)
            new_population.append(sim)
            
        population = new_population
        
    best_index = fit.index(max(fit))
    return population[best_index]