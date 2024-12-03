import torch
from gen import fitness, run_genetic_alg
from utils import vec2
from simulation import simulation


def init_torch():
    if(torch.cuda.is_available()):
        torch.set_default_device("cuda")
        print("Enabled cuda as default device")


def main():
    init_torch()

    sim = simulation(vec2(10, 10), 8, 10)
    sim.spawn_random_population()
    sim.grid.print()
    
    # Step 2: Start the genetic algorithm (adapted from my code)
    print("\nStarting Genetic Algorithm...")
    generations = 50  # Number of generations to evolve
    population_size = 10  # Size of population in each generation

    best_simulation = run_genetic_alg(generations, population_size)
    print("\nGenetic Algorithm complete!")
    print(f"Best simulation fitness: {fitness(best_simulation)}")
    
    
if __name__ == "__main__":
    main()


