# Ignore this: playground to test things

import torch 
import torch.nn
import torch.nn.functional
import gen
import organism
import simulation
from utils import vec2


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(4, 4)
        self.l2 = torch.nn.Linear(4, 4)
        self.l3 = torch.nn.Linear(4, 4)

    def forward(self, input):
        i1 = torch.nn.functional.relu(self.l1(input))
        i2 = torch.nn.functional.relu(self.l2(i1))
        o = torch.nn.functional.relu(self.l3(i2))
        return o

torch.set_default_device("cuda")
m = Model()

# Testing funcs:

gene1 = gen.generate_random_gene([4,4,4,4])
gene2 = gen.generate_random_gene([4,4,4,4])
crossed = gen.cross_genes(gene1, gene2)
mutated = gen.mutate_genes(gene1, 0.5)

sim = simulation.simulation(vec2(10,10), 50, 10)
sim.spawn_random_population()

# for org in sim.organism_list:
#     print(org.sickness)

sim.grid.print()

sim.organism_list[0].update_sickness()

# gen.run_genetic_alg(10, 10)
