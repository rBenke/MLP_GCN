import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv
from torch.nn import Linear
class MLP_GCN(torch.nn.Module):
    def __init__(self, architecture):
        super().__init__()
        n_mlp_layers = len(architecture)//2
        self.n_mlp_layers = n_mlp_layers
        self.layers = torch.nn.ModuleList()

        for layer_idx in range(n_mlp_layers):
            self.layers.append(Linear(architecture[layer_idx], architecture[layer_idx+1]))
        for layer_idx in range(n_mlp_layers, len(architecture)-1):
            self.layers.append(GCNConv(architecture[layer_idx], architecture[layer_idx+1]))

    def forward(self, x ,adj):
        for layer in self.layers[:-1]:
            if isinstance(layer, Linear):
                x = layer(x)
            else:
                x = layer(x,adj)
            x = F.relu(x)
        x = self.layers[-1](x, adj)

        return x
