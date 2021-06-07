import sys

import torch.nn as nn

from malib.model import TorchModel
from malib.model.Noisy import NoisyLinear
# from .torch_model import TorchModel


class MLP(TorchModel):
    def __init__(self, in_dim, out_dim, dueling=False, noisy=False, hidden_dim=256, device: str = "cpu") -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.noisy = noisy
        self.dueling = dueling


        if noisy:
            linear_layer = NoisyLinear
        else:
            linear_layer = nn.Linear
        self.noisy_layers = []

        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU()
        )

        linear_layer1 = linear_layer(self.hidden_dim, self.hidden_dim)
        linear_layer2 = linear_layer(self.hidden_dim, self.out_dim)
        
        self.q_layer = nn.Sequential(
            linear_layer1,
            nn.ReLU(),
            linear_layer2
        )
        if noisy:
            self.noisy_layers.append(linear_layer1)
            self.noisy_layers.append(linear_layer2)


        if dueling:
            linear_layer3 = linear_layer(self.hidden_dim, self.hidden_dim)
            linear_layer4 = linear_layer(self.hidden_dim, 1)
            self.value_layer = nn.Sequential(
                linear_layer3,
                nn.ReLU(), 
                linear_layer4
            )
            if noisy:
                self.noisy_layers.append(linear_layer3)
                self.noisy_layers.append(linear_layer4)

    def forward(self, x):
        feature = self.feature_layer(x)
        q = self.q_layer(feature)
        if self.dueling:
            value = self.value_layer(feature)
            q = value + q - q.mean(dim=-1, keepdim=True)
        return q

    def reset_noise(self):
        for noisy_layer in self.noisy_layers:
            noisy_layer.reset_noise()

if __name__ == "__main__":
    import torch

    a = torch.zeros((3, 12))
    m = MLP(12, 4,  noisy=True)
    m2 = MLP(12, 4, noisy=True)
    y = m(a)
    print(y.shape)
    print(m.noisy_layers)
    w = m.get_weights()
    m2.set_weights(w)
