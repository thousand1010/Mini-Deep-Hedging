"""
評価スクリプト
- 学習済み方策と解析デルタを比較
- 期末損益分布に加え、時間経過ごとの損益・ポジション・累積取引コストの推移を可視化
- 統計量・データフレームをファイルに保存
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

from src.data import gbm_simulate, bs_delta
from src.model import PolicyNet


def eval_model(model: torch.nn.Module, device: str, args, n_paths: int = 5000):
    """
    model: 学習済みモデル（PolicyNet）
    device: 'cpu' or 'cuda'
    args: argparse.Namespace 相当（S0, K, T, n_steps, mu, sigma, k, seed, r 等を期待）
    n_paths: 評価に使う経路数
    """
    # 価格経路生成
    S = gbm_simulate(args.S0, args.mu, args.sigma, args.T, args.n_steps, n_paths,
                     device=device, seed=args.seed)
    dt = args.T / args.n_steps

    model.eval()
    batch = n_paths
    prev_pos = torch.zeros(batch, device=device)
    cash_model = torch.zeros(batch, device=device)
    trade_costs_model = torch.zeros(batch, device=device)

    cash_delta = torch.zeros(batch, device=device)
    trade_costs_delta = torch.zeros(batch, device=device)
    prev_pos_delta = torch.zeros(batch, device=device)

    # 時系列ログ格納
    pl_model_ts, pl_delta_ts = [], []
    pos_model_ts, pos_delta_ts = [], []
    costs_model_ts, costs_delta_ts = [], []

    for t in range(args.n_steps + 1):
        if t < args.n_steps:
            time_to_maturity = (args.T - t * dt) * torch.ones(batch, device=device)
            obs = torch.stack([S[:, t], time_to_maturity, prev_pos], dim=1)

            # 学習モデルによるポジション
            pos = model(obs).detach()
            delta_pos = pos - prev_pos
            trade_costs_model += args.k * torch.abs(delta_pos) * S[:, t]
            cash_model -= delta_pos * S[:, t]
            prev_pos = pos

            # 解析デルタ
            tau = args.T - t * dt
            r_val = getattr(args, "r", 0.0)
            delta_bs = torch.tensor(bs_delta(S[:, t].cpu().numpy(),
                                             args.K, r_val, args.sigma, tau),
                                    device=device, dtype=torch.float32)
            delta_pos_bs = delta_bs - prev_pos_delta
            trade_costs_delta += args.k * torch.abs(delta_pos_bs) * S[:, t]
            cash_delta -= delta_pos_bs * S[:, t]
            prev_pos_delta = delta_bs
        else:
            # 最終清算
            cash_model += prev_pos * S[:, -1]
            cash_delta += prev_pos_delta * S[:, -1]

        # 途中経過 P&L を計算
        payoff_partial = torch.clamp(S[:, t] - args.K, min=0.0) if t < S.shape[1] else torch.clamp(S[:, -1] - args.K, min=0.0)
        pl_model_now = cash_model - payoff_partial - trade_costs_model
        pl_delta_now = cash_delta - payoff_partial - trade_costs_delta

        pl_model_ts.append(pl_model_now.cpu().numpy())
        pl_delta_ts.append(pl_delta_now.cpu().numpy())
        pos_model_ts.append(prev_pos.cpu().numpy())
        pos_delta_ts.append(prev_pos_delta.cpu().numpy())
        costs_model_ts.append(trade_costs_model.cpu().numpy())
        costs_delta_ts.append(trade_costs_delta.cpu().numpy())

    # numpy 配列化
    pl_model_ts = np.array(pl_model_ts)        # shape = (n_steps+1, n_paths)
    pl_delta_ts = np.array(pl_delta_ts)
    pos_model_ts = np.array(pos_model_ts)
    pos_delta_ts = np.array(pos_delta_ts)
    costs_model_ts = np.array(costs_model_ts)
    costs_delta_ts = np.array(costs_delta_ts)

    # 各時点で平均・中央値
    pl_model_ts_mean, pl_model_ts_median = pl_model_ts.mean(axis=1), np.median(pl_model_ts, axis=1)
    pl_delta_ts_mean, pl_delta_ts_median = pl_delta_ts.mean(axis=1), np.median(pl_delta_ts, axis=1)
    pos_model_ts_mean, pos_model_ts_median = pos_model_ts.mean(axis=1), np.median(pos_model_ts, axis=1)
    pos_delta_ts_mean, pos_delta_ts_median = pos_delta_ts.mean(axis=1), np.median(pos_delta_ts, axis=1)
    costs_model_ts_mean, costs_model_ts_median = costs_model_ts.mean(axis=1), np.median(costs_model_ts, axis=1)
    costs_delta_ts_mean, costs_delta_ts_median = costs_delta_ts.mean(axis=1), np.median(costs_delta_ts, axis=1)

    # 最終損益
    payoff = torch.clamp(S[:, -1] - args.K, min=0.0)
    terminal_pl_model = cash_model - payoff - trade_costs_model
    terminal_pl_delta = cash_delta - payoff - trade_costs_delta
    pl_model_result = terminal_pl_model.cpu().numpy()
    pl_delta_result = terminal_pl_delta.cpu().numpy()

    # データフレーム作成
    time_steps = np.arange(args.n_steps + 1)
    pl_ts_df = pd.DataFrame({
        'Time': np.tile(time_steps, 2),
        'Strategy': ['学習戦略'] * (args.n_steps + 1) + ['デルタヘッジ'] * (args.n_steps + 1),
        'Mean P&L': np.concatenate([pl_model_ts_mean, pl_delta_ts_mean]),
        'Median P&L': np.concatenate([pl_model_ts_median, pl_delta_ts_median])
    })

    pos_ts_df = pd.DataFrame({
        'Time': np.tile(time_steps, 2),
        'Strategy': ['学習戦略'] * (args.n_steps + 1) + ['デルタヘッジ'] * (args.n_steps + 1),
        'Mean Position': np.concatenate([pos_model_ts_mean, pos_delta_ts_mean]),
        'Median Position': np.concatenate([pos_model_ts_median, pos_delta_ts_median])
    })

    costs_ts_df = pd.DataFrame({
        'Time': np.tile(time_steps, 2),
        'Strategy': ['学習戦略'] * (args.n_steps + 1) + ['デルタヘッジ'] * (args.n_steps + 1),
        'Mean Cumulative Transaction Costs': np.concatenate([costs_model_ts_mean, costs_delta_ts_mean]),
        'Median Cumulative Transaction Costs': np.concatenate([costs_model_ts_median, costs_delta_ts_median])
    })

    terminal_pl_df = pd.DataFrame({
        'Strategy': ['学習戦略'] * len(pl_model_result) + ['デルタヘッジ'] * len(pl_delta_result),
        'Terminal P&L': np.concatenate([pl_model_result, pl_delta_result])
    })

    # 保存先ディレクトリ
    fig_dir = os.path.join(args.save_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # CSV 保存
    pl_ts_df.to_csv(os.path.join(fig_dir, "pl_time_series.csv"), index=False, encoding="utf-8-sig")
    pos_ts_df.to_csv(os.path.join(fig_dir, "position_time_series.csv"), index=False, encoding="utf-8-sig")
    costs_ts_df.to_csv(os.path.join(fig_dir, "costs_time_series.csv"), index=False, encoding="utf-8-sig")
    terminal_pl_df.to_csv(os.path.join(fig_dir, "terminal_pl_distribution.csv"), index=False, encoding="utf-8-sig")

    # 図の保存
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=pl_ts_df, x='Time', y='Mean P&L', hue='Strategy')
    plt.title('時間経過に伴う平均損益')
    plt.xlabel('時間ステップ'); plt.ylabel('平均損益'); plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "mean_pl_time_series.png"), bbox_inches="tight"); plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=pos_ts_df, x='Time', y='Mean Position', hue='Strategy')
    plt.title('時間経過に伴う平均ポジション')
    plt.xlabel('時間ステップ'); plt.ylabel('平均ポジション'); plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "mean_position_time_series.png"), bbox_inches="tight"); plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=costs_ts_df, x='Time', y='Mean Cumulative Transaction Costs', hue='Strategy')
    plt.title('時間経過に伴う平均累積取引コスト')
    plt.xlabel('時間ステップ'); plt.ylabel('平均累積取引コスト'); plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "mean_cumulative_costs.png"), bbox_inches="tight"); plt.close()

    plt.figure(figsize=(8, 6))
    sns.violinplot(data=terminal_pl_df, x='Strategy', y='Terminal P&L')
    plt.title('期末損益分布の比較')
    plt.xlabel('戦略'); plt.ylabel('期末損益'); plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "terminal_pl_distribution.png"), bbox_inches="tight"); plt.close()

    # 統計ファイル
    stats_path = os.path.join(fig_dir, "pl_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        for name, arr in [("学習戦略 (期末)", pl_model_result), ("デルタヘッジ (期末)", pl_delta_result)]:
            f.write(f"{name} 平均 {np.mean(arr):.4f}, 標準偏差 {np.std(arr):.4f}, 中央値 {np.median(arr):.4f}\n")

    print("図とCSV、統計を保存しました:", fig_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="results/policy_epoch30.pt")
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
    parser.add_argument("--save_dir", type=str, default="./results")
    args = parser.parse_args()

    device = args.device
    model = PolicyNet(obs_dim=3, hidden_sizes=[128, 128])
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    eval_model(model, device, args, n_paths=args.n_paths)
