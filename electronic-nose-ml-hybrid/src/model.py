import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, out_dim=2, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Sigmoid(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim))
    def forward(self, x):
        return self.net(x)
