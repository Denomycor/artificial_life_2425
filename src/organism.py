from simulation import simulation
from utils import vec2


# The class representing each living organism in the simulation
class organism:

    def __init__(self, sim: simulation, pos: vec2):
        self.sim = sim
        self.pos = pos
    
