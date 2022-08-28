from ogb.nodeproppred import PygNodePropPredDataset
import torch
from torch_geometric.loader import NeighborLoader

def load_data(dataset, batch_size):
    if dataset=="arxiv":
        dataset = PygNodePropPredDataset(name = "ogbn-arxiv",root="data/raw_data")
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph = dataset[0]

        train_dataloader = NeighborLoader(
            graph,
            num_neighbors=[-1]*2,
            batch_size=batch_size,
            input_nodes=train_idx,
            shuffle=True,
            num_workers=4
        )
        valid_dataloader = NeighborLoader(
            graph,
            num_neighbors=[-1]*2,
            batch_size=batch_size,
            input_nodes=valid_idx,
            shuffle=False,
            num_workers=4
        )
        test_dataloader = NeighborLoader(
            graph,
            num_neighbors=[-1]*2,
            batch_size=batch_size,
            input_nodes=test_idx,
            shuffle=False,
            num_workers=4
        )

        input_dim, output_dim = 128,40
    elif dataset=="":
        dataset = None
        (train_dataloader, valid_dataloader, test_dataloader), (input_dim, output_dim) = (0,0,0), (0,0)

    return (train_dataloader, valid_dataloader, test_dataloader), (input_dim, output_dim)
