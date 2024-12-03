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
    pass


# TODO: Create a new gene which is a mutation of the parameter gene
def mutate_genes(gene: OrderedDict) -> OrderedDict:
    pass


# TODO: From the population and it's fitness select pairs of organisms to generate offspring
def selection_tournament(population: list[simulation], fit: list[float]) -> list[tuple[organism, organism]]:
    pass


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
    population = start_genetic_alg(population_size)
    
    for _ in range(generations):
        fit = []
        for individual in population:
            individual.run()
            fit.append(fitness(individual))


