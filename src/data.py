"""
データ関連ユーティリティ（価格シミュレーション、Black-Scholes 価格/デルタ）
"""

import numpy as np
import torch
from scipy.stats import norm


def gbm_simulate(S0: float, mu: float, sigma: float, T: float, n_steps: int, n_paths: int,
                 device: str = "cpu", seed: int | None = None) -> torch.Tensor:
    """
    幾何ブラウン運動 (Geometric Brownian Motion, GBM) による価格経路シミュレーション（ベクトル化）
    出力: torch.Tensor of shape (n_paths, n_steps+1)

    パラメータ:
      S0: 初期価格
      mu: ドリフト（期待リターン）
      sigma: ボラティリティ
      T: 満期（年単位などの時間長）
      n_steps: 期間を分割するステップ数（再調整回数）
      n_paths: サンプル数（モンテカルロ経路数）
      device: 'cpu' または 'cuda'
      seed: 乱数シード
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    dt = T / n_steps
    # ドリフトとボラティリティ係数（対数差分の平均と標準偏差）
    drift = (mu - 0.5 * sigma ** 2) * dt
    vol = sigma * np.sqrt(dt)

    # 正規乱数 (n_paths, n_steps)
    z = torch.randn((n_paths, n_steps), device=device)
    increments = drift + vol * z            # 対数価格の増分 (log returns)

    # 対数価格の累積和をとって一括で価格経路を作成（高速化）
    logS = torch.cumsum(increments, dim=1)  # shape (n_paths, n_steps)
    S_tail = S0 * torch.exp(logS)           # shape (n_paths, n_steps)

    # 先頭に初期価格を挿入して (n_paths, n_steps+1) にする
    S = torch.zeros((n_paths, n_steps + 1), device=device)
    S[:, 0] = S0
    S[:, 1:] = S_tail

    return S


def bs_call_price(S, K: float, r: float, sigma: float, tau):
    """
    ブラック–ショールズ式による欧州コール価格（S は配列またはスカラー）
    tau: 残存期間（スカラー）
    ※ tau が 0 以下の場合はペイオフ max(S-K,0) を返す
    """
    S_arr = np.array(S)
    # tau が 0 以下なら期末ペイオフ
    if np.any(np.array(tau) <= 0):
        return np.maximum(S_arr - K, 0.0)

    d1 = (np.log(S_arr / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S_arr * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)


def bs_delta(S, K: float, r: float, sigma: float, tau):
    """
    ブラック–ショールズデルタ ∂C/∂S を返す
    """
    S_arr = np.array(S)
    if np.any(np.array(tau) <= 0):
        # 満期時のデルタはイン・ザ・マネーなら 1、そうでなければ 0
        return (S_arr > K).astype(float)

    d1 = (np.log(S_arr / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)