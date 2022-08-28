import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, architecure):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for layer_idx in range(len(architecure) - 1):
            self.layers.append(GCNConv(architecure[layer_idx], architecure[layer_idx + 1]))

    def forward(self, x, adj):
        for layer in self.layers[:-1]:
            x = layer(x, adj)
            x = F.relu(x)
        x = self.layers[-1](x, adj)

        return x
