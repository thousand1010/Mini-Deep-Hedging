import torch
import numpy as np
from src.data import gbm_simulate, bs_delta
from src.model import PolicyNet
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def eval_model(model, device, args, n_paths=5000):
    # 価格経路を生成
    S = gbm_simulate(args.S0, args.mu, args.sigma, args.T, args.n_steps, n_paths, device=device, seed=args.seed)
    dt = args.T / args.n_steps

    # 学習済みモデルとデルタヘッジの比較
    model.eval()
    batch = n_paths
    prev_pos = torch.zeros(batch, device=device)
    cash_model = torch.zeros(batch, device=device)
    trade_costs_model = torch.zeros(batch, device=device)

    cash_delta = torch.zeros(batch, device=device)
    trade_costs_delta = torch.zeros(batch, device=device)
    prev_pos_delta = torch.zeros(batch, device=device)

    for t in range(args.n_steps):
        time_to_maturity = (args.T - t*dt) * torch.ones(batch, device=device)
        obs = torch.stack([S[:,t], time_to_maturity, prev_pos], dim=1)
        # 学習モデルによるポジション
        pos = model(obs).detach()
        delta_pos = pos - prev_pos
        trade_costs_model += args.k * torch.abs(delta_pos) * S[:,t]
        cash_model -= delta_pos * S[:,t]
        prev_pos = pos

        # 解析的デルタヘッジ
        tau = args.T - t*dt
        r_val = getattr(args, "r", 0.0)  # args.r が無ければ 0.0 を使う
        delta_bs = torch.tensor(bs_delta(S[:,t].cpu().numpy(), args.K, args.r, args.sigma, tau), device=device, dtype=torch.float32)
        delta_pos_bs = delta_bs - prev_pos_delta
        trade_costs_delta += args.k * torch.abs(delta_pos_bs) * S[:,t]
        cash_delta -= delta_pos_bs * S[:,t]
        prev_pos_delta = delta_bs

    # 最後にポジションを清算
    cash_model += prev_pos * S[:, -1]
    cash_delta += prev_pos_delta * S[:, -1]

    # コール支払い
    payoff = torch.clamp(S[:, -1] - args.K, min=0.0)
    terminal_pl_model = cash_model - payoff - trade_costs_model
    terminal_pl_delta = cash_delta - payoff - trade_costs_delta

    # 結果を可視化
    pl_model = terminal_pl_model.cpu().numpy()
    pl_delta = terminal_pl_delta.cpu().numpy()

    sns.kdeplot(pl_model, label='学習戦略')
    sns.kdeplot(pl_delta, label='デルタヘッジ')
    plt.legend()
    plt.title('期末損益分布の比較')
    plt.show()

    # 基本統計量を表示
    for name, arr in [('学習戦略', pl_model), ('デルタヘッジ', pl_delta)]:
        print(name, "平均", np.mean(arr), "標準偏差", np.std(arr), "中央値", np.median(arr))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="results/policy_epoch50.pt")
    parser.add_argument("--device", type=str, default="cpu")
    # train と同じパラメータを再利用
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
    model = PolicyNet(obs_dim=3, hidden_sizes=[128,128])
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    eval_model(model, device, args, n_paths=args.n_paths)