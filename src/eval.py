"""
評価スクリプト
学習済み方策と解析デルタを比較し、期末損益分布を可視化・統計量を表示する。
"""

import torch
import numpy as np
from src.data import gbm_simulate, bs_delta
from src.model import PolicyNet
import matplotlib.pyplot as plt


def eval_model(model: torch.nn.Module, device: str, args, n_paths: int = 5000):
    """
    model: 学習済みモデル（PolicyNet）
    device: 'cpu' or 'cuda'
    args: argparse.Namespace 相当（S0, K, T, n_steps, mu, sigma, k, seed, r 等を期待）
    n_paths: 評価に使う経路数
    """
    # 価格経路生成
    S = gbm_simulate(args.S0, args.mu, args.sigma, args.T, args.n_steps, n_paths, device=device, seed=args.seed)
    dt = args.T / args.n_steps

    model.eval()
    batch = n_paths
    prev_pos = torch.zeros(batch, device=device)
    cash_model = torch.zeros(batch, device=device)
    trade_costs_model = torch.zeros(batch, device=device)

    cash_delta = torch.zeros(batch, device=device)
    trade_costs_delta = torch.zeros(batch, device=device)
    prev_pos_delta = torch.zeros(batch, device=device)

    for t in range(args.n_steps):
        time_to_maturity = (args.T - t * dt) * torch.ones(batch, device=device)
        obs = torch.stack([S[:, t], time_to_maturity, prev_pos], dim=1)

        # 学習モデルによるポジション
        pos = model(obs).detach()
        delta_pos = pos - prev_pos
        trade_costs_model += args.k * torch.abs(delta_pos) * S[:, t]
        cash_model -= delta_pos * S[:, t]
        prev_pos = pos

        # 解析デルタ（args.r が無ければデフォルト 0.0）
        tau = args.T - t * dt
        r_val = getattr(args, "r", 0.0)
        delta_bs = torch.tensor(bs_delta(S[:, t].cpu().numpy(), args.K, r_val, args.sigma, tau),
                                device=device, dtype=torch.float32)
        delta_pos_bs = delta_bs - prev_pos_delta
        trade_costs_delta += args.k * torch.abs(delta_pos_bs) * S[:, t]
        cash_delta -= delta_pos_bs * S[:, t]
        prev_pos_delta = delta_bs

    # 最終清算
    cash_model += prev_pos * S[:, -1]
    cash_delta += prev_pos_delta * S[:, -1]

    payoff = torch.clamp(S[:, -1] - args.K, min=0.0)
    terminal_pl_model = cash_model - payoff - trade_costs_model
    terminal_pl_delta = cash_delta - payoff - trade_costs_delta

    pl_model = terminal_pl_model.cpu().numpy()
    pl_delta = terminal_pl_delta.cpu().numpy()

    # 可視化
    try:
        import seaborn as sns
        sns.kdeplot(pl_model, label="学習戦略")
        sns.kdeplot(pl_delta, label="デルタヘッジ")
    except Exception:
        plt.hist(pl_model, bins=100, alpha=0.5, label="学習戦略", density=True)
        plt.hist(pl_delta, bins=100, alpha=0.5, label="デルタヘッジ", density=True)

    plt.legend()
    plt.title("期末損益分布の比較")
    plt.xlabel("Terminal P&L")
    plt.show()

    # 基本統計量の表示
    for name, arr in [("学習戦略", pl_model), ("デルタヘッジ", pl_delta)]:
        print(name, "平均", np.mean(arr), "標準偏差", np.std(arr), "中央値", np.median(arr))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="results/policy_epoch50.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--S0", type=float, default=100.0)
    parser.add_argument("--K", type=float, default=100.0)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--r", type=float, default=0.0)
    parser.add_argument("--k", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_paths", type=int, default=2000)
    args = parser.parse_args()

    device = args.device
    model = PolicyNet(obs_dim=3, hidden_sizes=[128, 128])
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    eval_model(model, device, args, n_paths=args.n_paths)