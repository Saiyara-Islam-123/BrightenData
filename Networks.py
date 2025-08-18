import torch.nn as nn

def cnn_flattened_output_size(vec_size, kernel, stride, padding, out_channels):
    return ((vec_size - kernel + (2 * padding))//stride + 1)*out_channels

def helper(vec_size):
    output_1 = cnn_flattened_output_size(vec_size, kernel=3, stride=2, padding=1, out_channels=4)
    output_2 = cnn_flattened_output_size(output_1, kernel=3, stride=2, padding=1, out_channels=8)
    return output_2

class VectorCNN(nn.Module):
    def __init__(self, vec_input_size, vec_output_size):
        self.conv_network = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, padding=1, stride=2),
                                          nn.ReLU(),
                                          nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1, stride=2),
                                          nn.ReLU(),
                                          nn.Flatten(),
                                          nn.Linear(helper(vec_input_size), vec_output_size)
                                        ) #I'll assume it's classification?

    def forward(self, x):
        return self.conv_network(x)

class RecurrentNetwork(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, nonlinearity, batch_size, output_size):
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity, batch_first=True)
        self.linear = nn.Linear(num_layers * hidden_size * input_size*batch_size, output_size)

    def forward(self, x):
        output, final_hidden_state =  self.rnn(x)
        return output