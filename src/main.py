import torch
import random
from gen import run_genetic_alg
from simulation import simulation
from utils import vec2


def init_torch():
    if(torch.cuda.is_available()):
        torch.set_default_device("cuda")
        print("Enabled cuda as default device")


def deterministic(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)


def main():
    init_torch()
    
    # Step 2: Start the genetic algorithm (adapted from my code)
    print("\nStarting Genetic Algorithm...")
    generations = 50  # Number of generations to evolve
    population_size = 10  # Size of population in each generation

    sim, fit = run_genetic_alg(generations, population_size)
    print("\nGenetic Algorithm complete!")
    if(sim != None):
        genes = sim.get_genes_of_organisms()
        print("Best fitness:", fit)
        print("Best Genes:", genes)

        print("Playing sim with winning genes:")
        winner_sim = simulation(vec2(20,20), 50, 20)
        winner_sim.spawn_random_population()
        winner_sim.set_genes_of_organisms(genes)
        winner_sim.grid.print()
        print()
        winner_sim.run(True)
    
    
if __name__ == "__main__":
    main()

