import os, sys
# src をインポートするため、プロジェクトルートを sys.path に追加
proj_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, proj_root)

import torch
import numpy as np
import matplotlib.pyplot as plt
from importlib import util

# モジュールのロード
def load_mod(path, name):
    spec = util.spec_from_file_location(name, path)
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

src_dir = os.path.join(proj_root, "src")
model_mod = load_mod(os.path.join(src_dir, "model.py"), "policy_model")
data_mod = load_mod(os.path.join(src_dir, "data.py"), "data_module")
PolicyNet = model_mod.PolicyNet
gbm_simulate = data_mod.gbm_simulate
bs_delta = data_mod.bs_delta

# 実験パラメータ
S0 = 100.0
K = 100.0
T = 1.0
# 時間分割
n_steps = 10        # 1経路あたりのタイムステップ数（再調整回数）
sigma = 0.2
mu = 0.0
k = 1e-3
# 学習時（ミニバッチ・反復）
batch_size = 64     # 1バッチあたりの経路数
epochs = 1
iters_per_epoch = 1 # 1エポックあたりのミニバッチ反復回数
lr = 1e-3

save_dir = os.path.join(proj_root, "results")
fig_dir = os.path.join(save_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

device = "cpu"
dt = T / n_steps

# シード値
torch.manual_seed(0); np.random.seed(0)

# モデル + オプティマイザ
model = PolicyNet(obs_dim=3, hidden_sizes=[128,128]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=lr)

# トレーニング
S = gbm_simulate(S0, mu, sigma, T, n_steps, batch_size, device=device, seed=0)
trade_costs = torch.zeros(batch_size, device=device)
cash = torch.zeros(batch_size, device=device)
prev_pos = torch.zeros(batch_size, device=device)
for t in range(n_steps):
    time_to_maturity = (T - t*dt) * torch.ones(batch_size, device=device)
    obs = torch.stack([S[:,t], time_to_maturity, prev_pos], dim=1)
    pos = model(obs)
    delta_pos = pos - prev_pos
    cost = k * torch.abs(delta_pos) * S[:,t]
    trade_costs += cost
    cash -= delta_pos * S[:,t]
    prev_pos = pos
cash += prev_pos * S[:,-1]
payoff = torch.clamp(S[:,-1] - K, min=0.0)
terminal_pl = cash - payoff - trade_costs
loss = torch.mean(terminal_pl**2)
opt.zero_grad(); loss.backward(); opt.step()

# モデルの保存
model_path = os.path.join(save_dir, "policy_tiny.pt")
torch.save(model.state_dict(), model_path)
print("Saved model to:", model_path)

# 評価
n_paths_eval = 500  # 評価で使う経路数（pl_distribution 用）
S_eval = gbm_simulate(S0, mu, sigma, T, n_steps, n_paths_eval, device=device)
model.eval()
prev_pos = torch.zeros(n_paths_eval, device=device)
cash_model = torch.zeros(n_paths_eval, device=device)
trade_costs_model = torch.zeros(n_paths_eval, device=device)
for t in range(n_steps):
    time_to_maturity = (T - t*dt) * torch.ones(n_paths_eval, device=device)
    obs = torch.stack([S_eval[:,t], time_to_maturity, prev_pos], dim=1)
    pos = model(obs).detach()
    delta_pos = pos - prev_pos
    trade_costs_model += k * torch.abs(delta_pos) * S_eval[:,t]
    cash_model -= delta_pos * S_eval[:,t]
    prev_pos = pos
cash_model += prev_pos * S_eval[:,-1]
payoff = torch.clamp(S_eval[:,-1] - K, min=0.0)
terminal_pl_model = cash_model - payoff - trade_costs_model
pl_model = terminal_pl_model.cpu().numpy()

# デルタベースラインの解析
prev_pos_delta = torch.zeros(n_paths_eval, device=device)
cash_delta = torch.zeros(n_paths_eval, device=device)
trade_costs_delta = torch.zeros(n_paths_eval, device=device)
for t in range(n_steps):
    tau = T - t*dt
    delta_bs_np = bs_delta(S_eval[:,t].cpu().numpy(), K, 0.0, sigma, tau)
    delta_bs = torch.tensor(delta_bs_np, device=device, dtype=torch.float32)
    delta_pos_bs = delta_bs - prev_pos_delta
    trade_costs_delta += k * torch.abs(delta_pos_bs) * S_eval[:,t]
    cash_delta -= delta_pos_bs * S_eval[:,t]
    prev_pos_delta = delta_bs
cash_delta += prev_pos_delta * S_eval[:,-1]
terminal_pl_delta = cash_delta - payoff - trade_costs_delta
pl_delta = terminal_pl_delta.cpu().numpy()

# 統計
import numpy as _np
def stats(arr):
    return {
        "平均": float(_np.mean(arr)),
        "標準偏差": float(_np.std(arr)),
        "中央値": float(_np.median(arr)),
        "5%分位": float(_np.quantile(arr, 0.05)),
        "95%分位": float(_np.quantile(arr, 0.95))
    }

print("学習戦略の統計:", stats(pl_model))
print("デルタヘッジの統計:", stats(pl_delta))

# プロットの保存
plt.figure(figsize=(6,3.5))
plt.hist(pl_model, bins=50, alpha=0.5, density=True, label='学習戦略')
plt.hist(pl_delta, bins=50, alpha=0.5, density=True, label='デルタヘッジ')
plt.legend()
plt.xlabel("期末損益（Terminal P&L）")
plt.title("期末損益分布の比較：学習戦略 vs デルタヘッジ")
save_path = os.path.join(fig_dir, "pl_tiny.png")
plt.savefig(save_path, bbox_inches="tight")
print("プロットを保存しました:", save_path)

