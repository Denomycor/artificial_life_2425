import torch
import torch.nn
import torch.nn.functional

from utils import vec2


# The class representing each living organism in the simulation
class organism:

    def __init__(self, sim, pos: vec2):
        self.sim = sim
        self.pos = pos
        self.neural = neural()
    

# Organism's ai
class neural(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(10,10)
        self.l2 = torch.nn.Linear(10,10)
        self.l3 = torch.nn.Linear(10,10)

    def forward(self, input):
        a1 = torch.nn.functional.relu(self.l1(input))
        a2 = torch.nn.functional.relu(self.l2(a1))
        out = torch.nn.functional.relu(self.l3(a2))
        return out

