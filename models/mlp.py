import torch
import torch.nn.functional as F
from torch.nn import Linear

class MLP(torch.nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for layer_idx in range(len(architecture) - 1):
            self.layers.append(Linear(architecture[layer_idx], architecture[layer_idx + 1]))

    def forward(self, x, adj):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
        x = self.layers[-1](x)
        return x
