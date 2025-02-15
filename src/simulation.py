from typing import OrderedDict
from organism import organism
from utils import vec2
from random import randint


# Main class for the simulation
class simulation:

    def __init__(self, grid_size: vec2, initial_pop: int, max_steps: int):
        self.step = 0
        self.max_steps = max_steps
        self.grid = grid(grid_size)
        self.initial_pop = initial_pop
        self.organism_list = []

    # Find random positions and genes for the initial population of the simulation
    def spawn_random_population(self):
        for _ in range(self.initial_pop):
            has_position = False
            while(not has_position):
                pos = vec2(randint(0, self.grid.size.x-1), randint(0, self.grid.size.y-1))
                if(not self.grid.has_organism(pos)):
                    has_position = True
                    # When created an automatic set of random genes is generated
                    org = organism(self, pos)
                    self.grid.set_organism(pos, org)
                    self.organism_list.append(org)
                    
    # For each organism set its genes to the provided ones
    def set_genes_of_organisms(self, genes: OrderedDict):
        for org in self.organism_list:
            org.set_neural_genes(genes)

    # Gets the genes that this simulation's organisms ar using
    def get_genes_of_organisms(self) -> OrderedDict:
        return self.organism_list[0].get_neural_genes()

    # Step the simulation once
    def run_step(self):
        for org in self.organism_list:
            org.process()


    # Run the simulation
    def run(self, display: bool = False):
        for _ in range(self.max_steps):
            self.run_step()
            if(display):
                self.grid.print()
                print()

    # Remove an organism from the simulation
    def erase_organism(self, organism):
        self.organism_list.remove(organism)
        self.grid.buffer[organism.pos] = None


# The 2D discrete space of the simulation
class grid:

    def __init__(self, size: vec2):
        self.size = size
        self._init_buffer()

    # Create and populate grid dictionary with null values
    def _init_buffer(self):
        self.buffer = {}
        for x in range(self.size.x):
            for y in range(self.size.y):
                self.set_organism(vec2(x,y), None)

    # Get the organism on grid at pos
    def get_organism(self, pos: vec2):
        return self.buffer[pos]

    # Set the organism on the grid at pos
    def set_organism(self, pos: vec2, value):
        self.buffer[pos] = value

    # Whether an organism exists at pos
    def has_organism(self, pos: vec2) -> bool:
        return self.get_organism(pos) != None

    # Move an organism from one pos to another 
    def move_organism(self, from_pos: vec2, to_pos: vec2):
        org = self.get_organism(from_pos)
        self.set_organism(from_pos, None)
        self.set_organism(to_pos, org)
        org.pos = to_pos
        
    # Check if the position is within the grid bounds
    def is_within_bounds(self, pos: vec2) -> bool:
        return 0 <= pos.x < self.size.x and 0 <= pos.y < self.size.y

    # Print the grid
    def print(self):
        for y in range(self.size.y):
            for x in range(self.size.x):
                print("1" if self.buffer[vec2(x,y)] else "0", end='')
            print()

