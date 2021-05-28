import sys

import torch.nn as nn

from malib.model import TorchModel


class AC(TorchModel):
    def __init__(self, in_dim, out_dim, device: str = "cpu") -> None:
        super(AC, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear1 = nn.Sequential(
            nn.Linear(self.in_dim, 256),
            nn.ReLU(),
        )
        self.policy = nn.Sequential(nn.Linear(256, self.out_dim[0]), nn.Softmax(dim=-1))
        self.v_linear = nn.Sequential(nn.Linear(256, self.out_dim[1]))

    def forward(self, x):
        x = self.linear1(x)
        x = self.policy(x)
        return x

    def v(self, x):
        x = self.linear1(x)
        x = self.v_linear(x)
        return x
