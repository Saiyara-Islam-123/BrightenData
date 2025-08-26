from Networks import *
from dataset import load_data
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os

def train_rnn(dataloader, lr, epochs, run):
    model = RecurrentNetwork(input_size=10, hidden_size=1, num_layers=2, nonlinearity="relu", batch_size=10, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    model.train()
    batch = 0
    loss_values = []

    for epoch in range(epochs):
        for x_seq, y_seq in dataloader:
            optimizer.zero_grad()
            outputs = model(x_seq.float())
            loss = criterion(outputs, y_seq.float())
            loss.backward()
            optimizer.step()
            print(epoch, loss.item())
            batch += 1
            loss_values.append(loss.item())
    os.mkdir(f"weights/RNN/{run}")
    torch.save(model.state_dict(), f"weights/RNN/{run}/{lr} final_weights.pth")
    d = {"loss": loss_values}
    df = pd.DataFrame(d)
    os.mkdir(f"losses/{run}")
    df.to_csv(f"losses/{run}/{lr} loss values RNN.csv", index=False)



if __name__ == '__main__':
    for i in range(100):
        train_loader, _ = load_data(batch_size=10)
        train_rnn(dataloader=train_loader, lr=0.01, epochs=10, run=i)