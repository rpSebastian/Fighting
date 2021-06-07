import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from malib.model import TorchModel
from malib.model.Noisy import NoisyLinear

class Categorical(TorchModel):
    def __init__(self, in_dim, out_dim, v_min,v_max, atom_size, noisy=False, dueling=False, hidden_dim=256, device: str = "cpu") -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.noisy = noisy
        self.dueling = dueling
        self.atom_size = atom_size
        self.support = torch.linspace(v_min, v_max, atom_size).to(device)

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
        linear_layer2 = linear_layer(self.hidden_dim, self.out_dim * self.atom_size)
        self.q_dist_layer = nn.Sequential(
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
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2).squeeze()

        if self.dueling:
            feature = self.feature_layer(x)
            value = self.value_layer(feature)
            q = value + q - q.mean(dim=-1, keepdim=True)

        return q

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        feature = self.feature_layer(x)
        q_atoms = self.q_dist_layer(feature).view(-1, self.out_dim, self.atom_size)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        return dist

    def reset_noise(self):
        for noisy_layer in self.noisy_layers:
            noisy_layer.reset_noise()

if __name__ == "__main__":
    import torch

    a = torch.zeros((12))
    m = Categorical(12, 4, 5, 10, 10, noisy=True, dueling=True)
    m2 = Categorical(12, 4, 5, 10, 10, noisy=True, dueling=True)
    y = m(a)
    print(m.noisy_layers)
    print(y.shape)
    w = m.get_weights()
    m2.set_weights(w)
