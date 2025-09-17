import torch.nn as nn

class Simple1DCNN(nn.Module):
    def __init__(self, in_len, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(1,16,5,1,2), nn.ReLU(), nn.MaxPool1d(2), nn.Conv1d(16,32,5,1,2), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(32, out_dim)
    def forward(self, x):
        x = x.unsqueeze(1); f = self.net(x).squeeze(-1); return self.head(f)

class SimpleGRU(nn.Module):
    def __init__(self, in_len, hidden=32, out_dim=2):
        super().__init__()
        self.rnn = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)
    def forward(self, x):
        x = x.unsqueeze(-1); out,_ = self.rnn(x); return self.head(out[:,-1,:])
