import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, hidden1_size=256, hidden2_size=128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(784, hidden1_size),
            nn.ReLU(),
            nn.Linear(hidden1_size, hidden2_size),
            nn.ReLU(),
            nn.Linear(hidden2_size, 10),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)