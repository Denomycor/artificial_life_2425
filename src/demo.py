# Ignore this: playground to test things

import torch 
import torch.nn
import torch.nn.functional
import gen 
import random

nn_arch = [4,5,6,1]

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(4, 5)
        self.l2 = torch.nn.Linear(5, 6)
        self.l3 = torch.nn.Linear(6, 1)

    def forward(self, input):
        i1 = torch.nn.functional.relu(self.l1(input))
        i2 = torch.nn.functional.relu(self.l2(i1))
        o = torch.nn.functional.relu(self.l3(i2))
        return o



torch.set_default_device("cuda")
m = Model()
# print(m.state_dict())
for k in m.state_dict():
    print(m.state_dict()[k].size())


out = gen.generate_random_gene(nn_arch)
for k in out:
    print(out[k].size())
