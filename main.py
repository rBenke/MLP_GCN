import argparse
from data.load_data import load_data
from models.create_model import create_model
from training import train
BATCH_SIZE = 128
NUM_EPOCHS =10

def train_validate_test(args=None):
    args = args if not None else parse_args()
    (tr_data, val_data, te_data), (input_dim, output_dim) = load_data(args.dataset, BATCH_SIZE)
    architecture = [input_dim] + [128,128] + [output_dim]
    model = create_model(args.model, architecture)
    print(model)
    train(model, tr_data, val_data, NUM_EPOCHS, BATCH_SIZE)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gcn", choices=['mlp', 'gcn', 'mlp_gcn'])
    parser.add_argument("--dataset", type=str, default="arxiv", choices=['arxiv', 'mag'])

    args = parser.parse_args()
    print(args)
    return args

if __name__=="__main__":
    train_validate_test()
