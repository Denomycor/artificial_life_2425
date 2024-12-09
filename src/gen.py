from functools import reduce
from typing import OrderedDict
from simulation import simulation
from utils import vec2
import torch


# Get the fitness value of a population
def fitness(sim) -> float:
    return reduce(lambda acc, e: acc + e.sickness, sim.organism_list, 0)


# Cross two different genes to create a new one
def cross_genes(st: OrderedDict, nd: OrderedDict) -> OrderedDict:
    child = OrderedDict()
    for key in st.keys():
        if torch.is_tensor(st[key]) and torch.is_tensor(nd[key]):
            # Perform element-wise crossover for tensors
            mask = torch.rand_like(st[key]) < 0.5
            child[key] = torch.where(mask, st[key], nd[key])
        else:
            # Randomly select gene from either parent
            if torch.rand(1).item() < 0.5:
                child[key] = st[key]
            else:
                child[key] = nd[key]
    return child


# Create a new gene which is a mutation of the parameter gene
# TODO: Ideally the mutation could happen on the tensor level and not on the key level
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


# From the population and it's fitness select pairs of organisms to generate offspring
def selection_tournament(population: list[simulation], fit: list[float], tournament_size: int, offspring_nr: int) -> list[tuple[simulation, simulation]]:
    selected_pairs = []
    for _ in range(offspring_nr):
        # Select first parent
        candidates = torch.randperm(len(population))[:tournament_size]
        best_candidate = min(candidates, key=lambda idx: fit[idx])
        parent1 = population[best_candidate]

        # Select second parent
        second_best_candidate = min(candidates, key=lambda idx: fit[idx] if fit[idx] > fit[best_candidate] else float('inf'))
        parent2 = population[second_best_candidate]

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
        sim.spawn_random_population()
        population.append(sim)
    return population


# TODO: Work in progress. Main loop of the genetic algorithm
def run_genetic_alg(generations: int, population_size: int):
    # Initialize the population
    population = start_genetic_alg(population_size)

    best_individual_ever = (None, float('inf'))
    
    for gen in range(generations):
        print(f"==Generation {gen + 1}/{generations}==")
        
        # Evaluate the fitness of the population
        fit: list[float] = []
        for individual in population:
            print(f"Running simulation {population.index(individual)}")
            individual.run()
            fit.append(fitness(individual))
        print(fit)
            
        for i in range(len(population)):
            # The best individual has the less fitness
            best_individual_ever = (population[i], fit[i]) if fit[i] < best_individual_ever[1] else best_individual_ever
        print(best_individual_ever[1])

        # Select parents for the next generation
        selected_pairs = selection_tournament(population, fit, len(population)//4, len(population)//2)
        
        # Crossover and mutation
        new_population = []
        for parent1, parent2 in selected_pairs:
            child_gene = cross_genes(parent1.get_genes_of_organisms(), parent2.get_genes_of_organisms())
            child_gene = mutate_genes(child_gene)
            
            # Create new simulation with the child gene
            sim = simulation(vec2(20,20), 50, 20)
            sim.spawn_random_population()
            sim.set_genes_of_organisms(child_gene)
            new_population.append(sim)

        # Fill the remaining individuals
        elite_count = population_size - len(new_population)
        if elite_count > 0:
            # Add the best individuals from the previous generation
            sorted_population = sorted(zip(population, fit), key=lambda x: x[1])
            elite = [ind for ind, _ in sorted_population[:elite_count]]
            new_population.extend(elite)
            
        population = new_population
        assert(len(population) == population_size)

