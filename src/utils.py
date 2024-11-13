import math


class vec2:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, rhs):
        return vec2(self.x + rhs.x, self.y + rhs.y)

    def __sub__(self, rhs):
        return vec2(self.x - rhs.x, self.y - rhs.y)

    def __mul__(self, e):
        return vec2(self.x * e, self.y * e)
    
    def __str__(self):
        return f"({self.x}, {self.y})"

    def __eq__(self, rhs) -> bool:
        return self.x == rhs.x and self.y == rhs.y 

    def __hash__(self):
        return hash((self.x, self.y))

    def len(self):
        return math.sqrt(math.pow(self.x, 2) + math.pow(self.y, 2))
    
    def distance_to(self, rhs):
        return abs((rhs - self).len())

