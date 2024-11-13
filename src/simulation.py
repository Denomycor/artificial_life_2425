from organism import organism
from utils import vec2
from random import randint


# Main class for the simulation
class simulation:

    def __init__(self, grid_size: vec2, initial_pop: int):
        self.sim_grid = sim_grid(grid_size)
        self.initial_pop = initial_pop

    # Find random positions for the initial population of the simulation
    def spawn_intial_population(self):
        for _ in range(self.initial_pop):
            has_position = False
            while(not has_position):
                pos = vec2(randint(0, self.sim_grid.size.x), randint(0, self.sim_grid.size.y))
                if(not self.sim_grid.has_organism(pos)):
                    has_position = True
                    org = organism(self, pos)
                    self.sim_grid.set_organism(pos, org)


# The 2D discrete space of the simulation
class sim_grid:

    def __init__(self, size: vec2):
        self.size = size
        self._init_grid()

    # Create and populate grid dictionary with null values
    def _init_grid(self):
        self.grid = {}
        for x in range(self.size.x):
            for y in range(self.size.y):
                self.set_organism(vec2(x,y), None)

    # Get the organism on grid at pos
    def get_organism(self, pos: vec2):
        return self.grid[pos]

    # Set the organism on the grid at pos
    def set_organism(self, pos: vec2, value):
        self.grid[pos] = value

    # Whether an organism exists at pos
    def has_organism(self, pos: vec2) -> bool:
        return self.get_organism(pos) != None

    # Move an organism from one pos to another 
    def move_organism(self, from_pos: vec2, to_pos: vec2):
        org = self.get_organism(from_pos)
        self.set_organism(from_pos, None)
        self.set_organism(to_pos, org)
        org.pos = to_pos

