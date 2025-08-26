from Networks import *
from dataset import load_data
import torch
import torch.nn as nn
import torch.optim as optim

def train_rnn(dataloader, lr, epochs):
    model = RecurrentNetwork(input_size=10, hidden_size=1, num_layers=2, nonlinearity="relu", batch_size=25, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    model.train()

    for epoch in range(epochs):
        for x_seq, y_seq in dataloader:
            optimizer.zero_grad()
            outputs = model(x_seq.float())

            loss = criterion(outputs, y_seq.float())
            loss.backward()
            optimizer.step()
            print(epoch, loss.item())



if __name__ == '__main__':
    train_loader, _ = load_data(batch_size=25)
    train_rnn(dataloader=train_loader, lr=0.01, epochs=10)