import torch

def train(model, data, epochs, batch_size):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):

        running_loss = 0.0
        for i, batch in enumerate(data):
            optimizer.zero_grad()
            prediction = model(batch.x, batch.edge_index)
            output = loss(prediction[:batch_size], batch.y[:batch_size, 0])
            output.backward()
            optimizer.step()

            running_loss += output.item()
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0