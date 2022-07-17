import torch
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList()

    def forward(self, x, adj):
        for layer in self.layers[:-1]:
            x = layer(x, adj)
            x = F.relu(x)
        x = self.layers[-1](x, adj)
        return x