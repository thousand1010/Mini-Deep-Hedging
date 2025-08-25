import torch
import torch.nn as nn

class PolicyNet(nn.Module):
    """
    単純な多層パーセプトロン (MLP)
    各時刻の観測量を入力として、保有すべき原資産の数量（実数）を出力する。
    入力 (batch, obs_dim) -> 出力 (batch,)
    """
    def __init__(self, obs_dim=3, hidden_sizes=[64,64], activation=nn.ReLU):
        super().__init__()
        layers = []
        dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(dim, h))
            layers.append(activation())
            dim = h
        layers.append(nn.Linear(dim, 1))    # 出力: 保有ポジション（任意の実数）
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)      # (batch,)