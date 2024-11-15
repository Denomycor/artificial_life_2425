# Ignore this: playground to test things

import torch 
import torch.nn
import torch.nn.functional



class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(4, 4)
        self.l2 = torch.nn.Linear(4, 4)
        self.l3 = torch.nn.Linear(4, 1)

    def forward(self, input):
        i1 = torch.nn.functional.relu(self.l1(input))
        i2 = torch.nn.functional.relu(self.l2(i1))
        o = torch.nn.functional.relu(self.l3(i2))
        return o

torch.set_default_device("cuda")
torch.manual_seed(41)
m = Model()
# m.to("cuda")
input = torch.rand(1,4)
print(m(input))
print(next(m.parameters()).device)
