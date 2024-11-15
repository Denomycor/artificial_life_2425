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
        self.l1 = torch.nn.Linear(1,1)

    def forward(self, input):
        out = torch.nn.functional.relu(self.l1(input))
        return out

