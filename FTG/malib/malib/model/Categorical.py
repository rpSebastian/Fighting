import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from malib.model import TorchModel

class Categorical(TorchModel):
    def __init__(self, in_dim, out_dim, v_min,v_max,atom_size,hidden_dim=256, device: str = "cpu") -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.atom_size = atom_size
        self.support = torch.linspace(v_min, v_max, atom_size).to(device)

        
        self.layers = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim), 
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), 
            nn.ReLU(), 
            nn.Linear(self.hidden_dim, self.out_dim * self.atom_size)
        )

    def forward(self, x):
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        
        return q

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        q_atoms = self.layers(x).view(-1, self.out_dim, self.atom_size)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist
    