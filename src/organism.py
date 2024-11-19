import torch
import torch.nn
import torch.nn.functional

from utils import vec2
from functools import reduce


# The class representing each living organism in the simulation
class organism:

    def __init__(self, sim, pos: vec2):
        self.sim = sim
        self.pos = pos
        self.sickness = 0
        self.neural = neural()
    
    # Called for each organism for each step of the simulation
    def process(self):
        #TODO: do sensor and action functions
        input = None
        out_mat: torch.Tensor = neural.forward(input)

    def get_neural_genes(self):
        return self.neural.state_dict()

    def set_neural_genes(self, genes):
        self.neural.load_state_dict(genes)

    """
    Organism sensors
    """

    def sensor_distance_nearest(self) -> float:
        organisms = filter(lambda e: e != self, self.sim.organism_list)
        distances = map(lambda e: self.pos.distance_to(e.pos), organisms)
        min_distance = reduce(min, distances, float('inf'))
        return min_distance

    def sensor_position_available(self, relative_pos: vec2) -> float:
        pos = self.pos + relative_pos
        return 1 if self.sim.grid.has_organism(pos) else 0

    def sensor_nearest_sickness(self) -> float:
        organisms = filter(lambda e: e != self, self.sim.organism_list)
        sickness = map(lambda e: e.sickness, organisms)
        min_sickness = reduce(min, sickness, float('inf'))
        return min_sickness


    """
    Organism actions
    """

    def action_move_to(self, relative_pos: vec2):
        pos = self.pos + relative_pos
        if not self.sim.grid.has_organism(pos):
            self.sim.grid.buffer[self.pos] = None
            self.sim.grid.buffer[pos] = self
            self.pos = pos


# Organism's ai
class neural(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(10,10)
        self.l2 = torch.nn.Linear(10,10)
        self.l3 = torch.nn.Linear(10,10)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        a1 = torch.nn.functional.relu(self.l1(input))
        a2 = torch.nn.functional.relu(self.l2(a1))
        out = torch.nn.functional.relu(self.l3(a2))
        return out

