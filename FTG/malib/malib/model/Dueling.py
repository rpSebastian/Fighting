import sys

import torch.nn as nn

from malib.model import TorchModel

# from .torch_model import TorchModel


class Dueling(TorchModel):
    def __init__(self, in_dim, out_dim, hidden_dim=256, device: str = "cpu") -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.advantage_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, out_dim)
        )

        self.value_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, x):
        feature = self.feature_layer(x)
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


if __name__ == "__main__":
    import torch
    a = torch.zeros((3, 12))
    m = Dueling(12, 4)
    m2 = Dueling(12, 4)
    y = m(a)
    print(y.shape)
    w = m.get_weights()
    m2.set_weights(w)
