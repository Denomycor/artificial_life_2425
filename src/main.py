import torch
import random
from gen import run_genetic_alg


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
        print("Best fitness:", fit)
        print("Best Genes:", sim.get_genes_of_organisms())
    
    
if __name__ == "__main__":
    main()

