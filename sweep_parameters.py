N_LAYERS = [3]#,6]
DATASETS = ["arxiv"]#, "products", "papers100M", "mag"]
GNN_LAYER = ["GCN"]#, "GraphSage", "GAT", "GIN"]
GNN_DENSE_SPARSE_SPLIT = [False]

PARAMETER_SPACE = [N_LAYERS, DATASETS, GNN_LAYER, GNN_DENSE_SPARSE_SPLIT]
PARAMETER_NAMES = ["n_layers", "dataset","gnn_layer", "gnn_dense_sparse_split"]
