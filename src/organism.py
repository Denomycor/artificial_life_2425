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
        input_data = [
            self.sensor_distance_nearest(),                 # Distance to nearest organism
            self.sensor_position_available(vec2(0, 1)),    # Is up available?
            self.sensor_position_available(vec2(0, -1)),   # Is down available?
            self.sensor_position_available(vec2(-1, 0)),   # Is left available?
            self.sensor_position_available(vec2(1, 0)),    # Is right available?
            self.sensor_nearest_sickness(),                # Sickness level of nearest organism
        ]
        
        input_tensor = torch.tensor(input_data)

        output_tensor = self.neural.forward(input_tensor)
        
        directions = [
            vec2(0, 1),   # Up
            vec2(0, -1),  # Down
            vec2(-1, 0),  # Left
            vec2(1, 0),   # Right
        ]   
        
        maxed = 0 
        for i in range(4):
            maxed = i if output_tensor[i].item() > output_tensor[maxed].item() else maxed

        self.action_move_to(directions[maxed])
        

    def get_neural_genes(self):
        return self.neural.state_dict()

    def set_neural_genes(self, genes):
        self.neural.load_state_dict(genes, False)

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
        if not self.sim.grid.is_within_bounds(pos):
            return 0
        return 1 if not self.sim.grid.has_organism(pos) else 0

    def sensor_nearest_sickness(self) -> float:
        organisms = filter(lambda e: e != self, self.sim.organism_list)
        nearest_organism = min(organisms, key=lambda e: self.pos.distance_to(e.pos), default=None)
        return nearest_organism.sickness if nearest_organism else float('inf')


    """
    Organism actions
    """

    def action_move_to(self, relative_pos: vec2):
        pos = self.pos + relative_pos
        if self.sensor_position_available(relative_pos):
            self.sim.grid.move_organism(self.pos, pos)
            self.pos = pos


# Organism's ai
class neural(torch.nn.Module):
    
    arch = [6,10,10,4]

    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(6,10)
        self.l2 = torch.nn.Linear(10,10)
        self.l3 = torch.nn.Linear(10,4)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        a1 = torch.nn.functional.relu(self.l1(input))
        a2 = torch.nn.functional.relu(self.l2(a1))
        out = self.l3(a2)
        return out

