import torch
import torch.nn.functional as F
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.conv import GCNConv, GATConv, GINConv, SAGEConv

def get_gnn_layer_cls(name):
    name = name.lower()
    if name=="gcn":
        return GCNConv
    elif name=="gat":
        return GATConv
    elif name=="graphsage":
        return SAGEConv
    elif name=="gin":
        return GINConv
    else:
        raise Exception(f"{name.upper()} is not implemented.")
        

class Mixed_model(torch.nn.Module):
    def __init__(self, gnn_layer_name, architecture):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.GNN_layer_cls = get_gnn_layer_cls(gnn_layer_name)
        print(architecture)
        for layer in architecture:
            if layer["name"]==gnn_layer_name:
                self.layers.append(self.GNN_layer_cls(layer["input_dim"], layer["output_dim"]))
            else:
                self.layers.append(Linear(layer["input_dim"], layer["output_dim"]))

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, self.GNN_layer_cls):
                x = layer(x, adj)
            else:
                x = layer(x)
            if i<len(self.layers)-1:
                x = F.relu(x)
        return x

