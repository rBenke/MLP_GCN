import torch
import numpy as np
from tqdm import tqdm
from ogb.nodeproppred import Evaluator

def train(model, train_data, valid_data, epochs, batch_size):
    model = model.to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = torch.nn.CrossEntropyLoss()
    previous_valid_loss = np.inf  
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_loader_iter = tqdm(enumerate(train_data))
        for i, batch in train_loader_iter:
            batch = batch.to("cuda") 
            optimizer.zero_grad()
            prediction = model(batch.x, batch.edge_index)
            output = loss(prediction[:batch_size], batch.y[:batch_size, 0])
            output.backward()
            optimizer.step()

            running_loss += output.item()
            train_loader_iter.set_description(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss/(i+1):.5f}')

        # validation metrics
        model.eval()
        valid_loss_sum = 0
        for i, batch in tqdm(enumerate(valid_data)):
            prediction = model(batch.x, batch.edge_index)
            valid_loss = loss(prediction[:batch_size], batch.y[:batch_size, 0])
            valid_loss_sum += valid_loss.item()
        # early stopping
        if valid_loss_sum > previous_valid_loss:
            break
        else:
            previous_valid_loss = valid_loss_sum
    print("Best valid loss: ", previous_valid_loss/len(valid_data))
    

def evaluate(model,train_data, valid_data, test_data, model_name):
    model.eval()
    evaluator = Evaluator(name=model_name)
    output = []
    for data in [train_data, valid_data, test_data]:
        predicted = []
        correct = []
        for i, batch in tqdm(enumerate(test_data)):
            prediction = model(batch.x, batch.edge_index)
            predicted.append(torch.round(prediction[:batch_size]).to(torch.int32))
            correct.append(batch.y[:batch_size, 0])
        correct = torch.concat(correct, axis=0)
        predicted = torch.concat(predicted, axis=0)
        output.append(evaluator({"y_true":correct, "y_pred":predicted}))

    return dict(zip(["train","valid", "test"], output))

