import argparse
import torch
import torch.optim as optim
from src.model import PolicyNet
from src.data import gbm_simulate
import os
from tqdm import tqdm

def payoff_call(S_T, K):
    """ヨーロピアンコールのペイオフ max(S_T - K, 0)"""
    return torch.clamp(S_T - K, min=0.0)

def train(args):
    # デバイス選択
    device = 'cuda' if (args.use_gpu and torch.cuda.is_available()) else 'cpu'
    torch.manual_seed(args.seed)

    # モデル定義
    obs_dim = 3                                                 # 入力は [S_t, 残存期間, 直前のポジション]
    model = PolicyNet(obs_dim=obs_dim, hidden_sizes=[128,128]).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    n_steps = args.n_steps
    dt = args.T / n_steps

    for epoch in range(args.epochs):
        model.train()
        losses = []
        pbar = tqdm(range(args.iters_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs}")
        for _ in pbar:
            # 価格経路をバッチで生成
            S = gbm_simulate(args.S0, args.mu, args.sigma, args.T, n_steps, args.batch_size, device=device)
            positions = torch.zeros((args.batch_size, n_steps+1), device=device)
            trade_costs = torch.zeros(args.batch_size, device=device)
            cash = torch.zeros(args.batch_size, device=device)  # 取引による現金
            prev_pos = torch.zeros(args.batch_size, device=device)

            # 各時刻でのポジション決定と取引コスト計算
            for t in range(n_steps):
                time_to_maturity = (args.T - t*dt) * torch.ones(args.batch_size, device=device)
                obs = torch.stack([S[:,t], time_to_maturity, prev_pos], dim=1)
                pos = model(obs)                                # ネットワークが出力する保有量
                delta_pos = pos - prev_pos
                # 取引コスト: k * |Δpos| * S_t
                cost = args.k * torch.abs(delta_pos) * S[:,t]
                trade_costs += cost
                # 現金変動: 購入ならマイナス、売却ならプラス
                cash -= delta_pos * S[:,t]
                positions[:, t] = pos
                prev_pos = pos

            # 期末でポジションを清算（すべて売却）
            cash += prev_pos * S[:, -1]
            positions[:, -1] = prev_pos

            # オプションの支払い（コールをショートしている）
            payoff = payoff_call(S[:, -1], args.K)
            # 最終損益
            terminal_pl = cash - payoff - trade_costs

            # 損失関数: 期末PLの二乗平均
            loss = torch.mean(terminal_pl**2)
            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=float(sum(losses)/len(losses)))
        print(f"Epoch {epoch+1} 平均損失: {sum(losses)/len(losses):.6f}")
        
        # モデルの保存
        if (epoch+1) % args.save_every == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"policy_epoch{epoch+1}.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--S0", type=float, default=100.0)
    parser.add_argument("--K", type=float, default=100.0)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--k", type=float, default=1e-3)        # 取引コスト係数
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--iters_per_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--r", type=float, default=0.0, help="risk-free interest rate")


    args = parser.parse_args()
    train(args)
