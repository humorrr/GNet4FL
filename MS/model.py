import torch
import torch.nn as nn
import torch.nn.functional as F

# from dgl.nn.pytorch import GATConv

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=100):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 16, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        return w
