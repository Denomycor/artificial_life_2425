import torch
from gen import run_genetic_alg


def init_torch():
    if(torch.cuda.is_available()):
        torch.set_default_device("cuda")
        print("Enabled cuda as default device")


def main():
    init_torch()
    
    # Step 2: Start the genetic algorithm (adapted from my code)
    print("\nStarting Genetic Algorithm...")
    generations = 50  # Number of generations to evolve
    population_size = 10  # Size of population in each generation

    run_genetic_alg(generations, population_size)
    print("\nGenetic Algorithm complete!")
    
    
if __name__ == "__main__":
    main()

