import torch.nn as nn
import torch.nn.functional as F
import torch

class BasicDistributionNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(BasicDistributionNetwork, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, 64)
        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.constant(self.fc1.bias, 0)

        self.fc2 = nn.Linear(64, 64)
        nn.init.xavier_uniform(self.fc2.weight)
        nn.init.constant(self.fc2.bias, 0)

        self.mean = nn.Linear(64, output_size)
        nn.init.xavier_uniform(self.mean.weight)
        nn.init.constant(self.mean.bias, 0)

        self.action_log_std = nn.Parameter(torch.zeros(1, output_size))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.mean(x)

        return x, self.action_log_std, torch.exp(self.action_log_std)


class ValueNetwork(nn.Module):
    def __init__(self, input_size):
        super(ValueNetwork, self).__init__()

        self.input_size = input_size

        self.fc1 = nn.Linear(input_size, 64)
        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.constant(self.fc1.bias, 0)

        self.fc2 = nn.Linear(64, 64)
        nn.init.xavier_uniform(self.fc2.weight)
        nn.init.constant(self.fc2.bias, 0)

        self.output = nn.Linear(64, 1)
        nn.init.xavier_uniform(self.output.weight)
        nn.init.constant(self.output.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)

        return x