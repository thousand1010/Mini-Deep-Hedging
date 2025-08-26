"""
方策ネットワーク（PolicyNet）
各時刻の観測量を入力として、保有すべき原資産の数量を出力する単純なMLP。
"""

import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    """
    シンプルな MLP 方策ネットワーク。

    入力例: [S_t, time_to_maturity, prev_pos]
    出力: 実数（その時点で保有する原資産の数量）
    """
    def __init__(self, obs_dim: int = 3, hidden_sizes: list[int] = [128, 128], activation=nn.ReLU):
        super().__init__()
        layers = []
        dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(dim, h))
            layers.append(activation())
            dim = h
        layers.append(nn.Linear(dim, 1))  # 出力: ポジション量（スカラー）
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, obs_dim) -> 出力 (batch,)
        """
        return self.net(x).squeeze(-1)