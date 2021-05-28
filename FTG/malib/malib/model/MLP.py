import sys

import torch.nn as nn

from malib.model import TorchModel

# from .torch_model import TorchModel


class MLP(TorchModel):
    def __init__(self, in_dim, out_dim, hidden_dim=256, device: str = "cpu") -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.linear1 = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.linear1_2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )

        self.linear2 = nn.Sequential(nn.Linear(self.hidden_dim, self.out_dim))

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear1_2(x)
        x = self.linear2(x)
        return x


if __name__ == "__main__":
    import torch

    a = torch.zeros((3, 12))
    m = MLP(12, 4)
    m2 = MLP(12, 4)
    y = m(a)
    print(y.shape)
    w = m.get_weights()
    m2.set_weights(w)
