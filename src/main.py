import torch
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


if __name__ == "__main__":
    main()

