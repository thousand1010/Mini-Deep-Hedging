import numpy as np
import torch
from scipy.stats import norm
import numpy as np

def gbm_simulate(S0, mu, sigma, T, n_steps, n_paths, device='cpu', seed=None):
    """
    幾何ブラウン運動 (Geometric Brownian Motion, GBM) による価格経路シミュレーション
    戻り値は (n_paths, n_steps+1) 形状のテンソル
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    dt = T / n_steps
    # ドリフト項とボラティリティ項
    drift = (mu - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    # 正規乱数を生成
    z = torch.randn((n_paths, n_steps), device=device)
    increments = drift + vol * z
    S = torch.zeros((n_paths, n_steps+1), device=device)
    S[:,0] = S0
    S[:,1:] = (S0 * torch.exp(torch.cumsum(increments, dim=1))).clone()
    # 経路を逐次計算
    for i in range(1, n_steps+1):
        S[:, i] = S[:, i-1] * torch.exp(increments[:, i-1])
    return S

def bs_call_price(S, K, r, sigma, tau):
    """
    ブラック–ショールズ式による欧州コールの価格
    tau: 残存期間
    """
    if tau <= 0:
        return np.maximum(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)

def bs_delta(S, K, r, sigma, tau):
    """
    ブラック–ショールズ式に基づくデルタ（∂V/∂S）
    """
    if tau <= 0:
        return (S > K).astype(float)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)