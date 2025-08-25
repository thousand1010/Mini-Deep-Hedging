# Mini-Deep-Hedging
A minimal implementation of Deep Hedging (Bühler et al. 2019) in PyTorch. Compares learned strategies with classical delta hedging under transaction costs.

PyTorchを用いたDeep Hedging（Bühler et al. 2019）の最小限の実装をする。取引コスト下で学習された戦略と古典的なデルタヘッジングを比較する。

**目的**  
ニューラルネットワークを用いてヨーロピアンコールのヘッジ戦略（ディスクリート再調整）を学習し、従来のデルタヘッジと比較する。取引コスト（比例コスト）を考慮し、最終のヘッジ損益（P&L）を最小化することを目標とする。

**Key idea**

* シミュレーション（GBM）で株価経路を作り、ネットワークが各時刻での保有量（underlying の数量）を出力する。
* 端末時点での P\&L（オプションの支払＋ヘッジによる収益－取引コスト）の二乗平均を損失にして最小化する。

## References

- Bühler, H., Gonon, L., Teichmann, J., & Wood, B. (2019).
  *Deep Hedging*. Quantitative Finance, 19(8), 1271–1291.  
  https://doi.org/10.1080/14697688.2019.1571683

- Black, F., & Scholes, M. (1973).  
  The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*, 81(3), 637–654.

- Hull, J. C. (2003).
  *フィナンシャルエンジニアリング : デリバティブ取引とリスク管理の総体系*（田渕直也 監訳）. ピアソン・エデュケーション.



