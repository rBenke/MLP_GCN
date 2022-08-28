import pandas as pd
import itertools

from const import BATCH_SIZE, NUM_EPOCHS
from sweep_parameters import PARAMETER_SPACE, PARAMETER_NAMES
from training import train, evaluate
from data.load_data import load_data
from models.create_model import create_mixed_model
def get_parameters():
    iter_param_space = itertools.product(*PARAMETER_SPACE)
    for params in iter_param_space:
        exp_params = {
                PARAMETER_NAMES[0]: params[0], 
                PARAMETER_NAMES[1]: params[1],
                PARAMETER_NAMES[2]: params[2],
                PARAMETER_NAMES[3]: params[3]}
        iter_dense_sparse_mix = itertools.product(
                [exp_params["gnn_layer"], "linear"], repeat=exp_params["n_layers"])
        for architecture in iter_dense_sparse_mix:
            exp_params["architecture"] = architecture
            yield exp_params

def get_experiment_name():
    return "test_exp"

def train_validate_test(params):
    (tr_data, val_data, te_data), (input_dim, output_dim) = load_data(params["dataset"], BATCH_SIZE)
    architecture = [input_dim]*(params["n_layers"])+[output_dim]
    architecture_dict = [
            {"name": params["architecture"], "input_dim":architecture[i], "output_dim":architecture[i+1]}
            for i in range(params["n_layers"]) ]
    model = create_mixed_model(params["gnn_layer"], architecture_dict)
    #print(model)
    train(model, tr_data, val_data, NUM_EPOCHS, BATCH_SIZE)
    results = evaluate(model, tr_data,val_data,te_data, BATCH_SIZE, params["dataset"])
    return {"results":results, "arch":architecture_dict}

def run_sweep():
    exp_name = get_experiment_name()
    results = pd.DataFrame(columns=["n_layers", "model", "architecture", "train_loss", "test_loss"])
    for param in get_parameters():
        metrics = train_validate_test(param)
        metrics = {
                "n_layers":param["n_layers"],
                "model":param["model"],
                "architecture":str(metrics["arch"]),
                "train_loss": metrics["results"]["train"],
                "test_loss": metrics["results"]["test"]
        }
        results.append(metrics)
        results.to_csv("/data/sweep_results/"+exp_name+".csv")

if __name__=="__main__":
    run_sweep()

